"""Panel schema shared by all data sources."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

PANEL_SCHEMA = pa.DataFrameSchema(
    {
        "unique_id": pa.Column(str),
        "ds": pa.Column(pa.DateTime),
        "y": pa.Column(float, pa.Check.ge(0)),
        "price": pa.Column(float, pa.Check.gt(0)),
        "promo": pa.Column(int, pa.Check.isin([0, 1])),
    },
    unique=["unique_id", "ds"],
    coerce=True,
)


def validate_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort a panel. Raises ``pandera.errors.SchemaError`` on violations."""
    if df.duplicated(["unique_id", "ds"]).any():
        raise ValueError("duplicate (unique_id, ds) rows")
    out = PANEL_SCHEMA.validate(df)
    out = out.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    counts = out.groupby("unique_id")["ds"].agg(["min", "max", "count"])
    expected = (counts["max"] - counts["min"]).dt.days + 1
    gaps = counts.index[counts["count"] != expected]
    if len(gaps):
        raise ValueError(f"series with missing days: {list(gaps[:5])}")
    return out

"""Common forecaster interface.

Every model receives the history up to the forecast origin and the *known future covariates*
(planned price and promotion calendar) and returns one row per series and future day.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

KEYS = ["unique_id", "ds"]


def add_price_rel(train: pd.DataFrame, future: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add ``price_rel`` = price / median training price of the series (scale-free covariate)."""
    ref = train.groupby("unique_id")["price"].median().rename("price_ref")
    train = train.join(ref, on="unique_id")
    future = future.join(ref, on="unique_id")
    train = train.assign(price_rel=train["price"] / train["price_ref"]).drop(columns="price_ref")
    future = future.assign(price_rel=future["price"] / future["price_ref"]).drop(
        columns="price_ref"
    )
    return train, future


def horizon(future: pd.DataFrame) -> int:
    return int(future.groupby("unique_id").size().max())


class Forecaster(ABC):
    """Returns columns ``unique_id, ds, yhat`` and optionally ``q10, q90`` (native quantiles)."""

    name: str

    @abstractmethod
    def forecast(self, train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame: ...

    @staticmethod
    def _finish(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
        out = df.rename(columns=cols)
        for c in cols.values():
            out[c] = np.clip(out[c].to_numpy(dtype=float), 0.0, None)
        return out[KEYS + list(cols.values())].reset_index(drop=True)

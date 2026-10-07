"""Loader for the M5 (Walmart) dataset.

The files cannot be downloaded automatically (Kaggle login required). Download them from
https://www.kaggle.com/competitions/m5-forecasting-accuracy/data and place these three files in one
directory: ``sales_train_evaluation.csv``, ``calendar.csv``, ``sell_prices.csv``.

``promo`` is not part of M5. It is a proxy: the week's price is at least 5% below the item's highest
price in the preceding 8 weeks. Treat results on this proxy accordingly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fdl.data.schema import validate_panel


def load_m5(
    directory: str | Path,
    n_series: int = 200,
    min_mean_sales: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    d = Path(directory)
    sales = pd.read_csv(d / "sales_train_evaluation.csv")
    calendar = pd.read_csv(d / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(d / "sell_prices.csv")

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    keep = sales[day_cols].mean(axis=1) >= min_mean_sales
    sales = sales.loc[keep]
    if len(sales) > n_series:
        sales = sales.sample(n=n_series, random_state=seed)
    sales = sales.assign(unique_id=sales["item_id"] + "_" + sales["store_id"])

    long = sales.melt(
        id_vars=["unique_id", "item_id", "store_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="y",
    )
    long = long.merge(calendar[["d", "date", "wm_yr_wk"]], on="d", how="left")
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long = long.rename(columns={"date": "ds", "sell_price": "price"})
    long = long.sort_values(["unique_id", "ds"])
    # M5 has no price before the item's first sale week; fill backwards inside each series
    long["price"] = long.groupby("unique_id")["price"].transform(lambda s: s.bfill().ffill())
    long = long.dropna(subset=["price"])

    weekly_max = long.groupby("unique_id")["price"].transform(
        lambda s: s.rolling(8 * 7, min_periods=1).max().shift(1)
    )
    long["promo"] = (long["price"] < 0.95 * weekly_max.fillna(long["price"])).astype(int)
    panel = long[["unique_id", "ds", "y", "price", "promo"]].astype(
        {"y": float, "price": float, "promo": int}
    )
    return validate_panel(panel)

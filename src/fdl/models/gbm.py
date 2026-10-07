"""Feature-based global gradient boosting (LightGBM through mlforecast).

One model is trained across all series. Besides the mean forecast (Tweedie objective, suited to
non-negative count demand) it fits two native quantile models, used to compare against conformal
intervals.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean

from fdl.models.base import KEYS, Forecaster, add_price_rel, horizon


def build_mlforecast(seed: int = 0, n_estimators: int = 300, quantiles: bool = True) -> MLForecast:
    common = dict(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )
    models: dict[str, lgb.LGBMRegressor] = {
        "yhat": lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2, **common)
    }
    if quantiles:
        models["q10"] = lgb.LGBMRegressor(objective="quantile", alpha=0.1, **common)
        models["q90"] = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **common)
    return MLForecast(
        models=models,
        freq="D",
        lags=[1, 2, 7, 14, 21, 28],
        lag_transforms={1: [RollingMean(7), RollingMean(28)], 7: [RollingMean(4)]},
        date_features=["dayofweek", "month"],
        num_threads=1,
    )


class LightGBMForecaster(Forecaster):
    name = "lightgbm"

    def __init__(self, seed: int = 0, n_estimators: int = 300, quantiles: bool = True) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.quantiles = quantiles

    def forecast(self, train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        train, future = add_price_rel(train, future)
        cols = ["price_rel", "promo"]
        fc = build_mlforecast(self.seed, self.n_estimators, self.quantiles)
        fc.fit(train[KEYS + ["y"] + cols], static_features=[])
        out = fc.predict(horizon(future), X_df=future[KEYS + cols])
        names = {"yhat": "yhat", **({"q10": "q10", "q90": "q90"} if self.quantiles else {})}
        out = self._finish(out, names)
        if self.quantiles:  # quantile crossing: sort per row
            lo = out[["q10", "yhat", "q90"]].min(axis=1)
            hi = out[["q10", "yhat", "q90"]].max(axis=1)
            out["q10"], out["q90"] = lo, hi
        return out

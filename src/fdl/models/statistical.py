"""Classical statistical forecasters (statsforecast)."""

from __future__ import annotations

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive

from fdl.models.base import KEYS, Forecaster, add_price_rel, horizon


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the value from the same weekday one week ago. The baseline every model must beat."""

    name = "seasonal_naive"

    def forecast(self, train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        sf = StatsForecast(models=[SeasonalNaive(season_length=7, alias="yhat")], freq="D")
        out = sf.forecast(df=train[KEYS + ["y"]], h=horizon(future))
        return self._finish(out, {"yhat": "yhat"})


class ETSForecaster(Forecaster):
    """Exponential smoothing with automatic model selection (AICc). No covariates."""

    name = "ets"

    def forecast(self, train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        sf = StatsForecast(models=[AutoETS(season_length=7, alias="yhat")], freq="D")
        out = sf.forecast(df=train[KEYS + ["y"]], h=horizon(future))
        return self._finish(out, {"yhat": "yhat"})


class ARIMAForecaster(Forecaster):
    """Seasonal ARIMA with automatic order selection and regression on price and promotion."""

    name = "arima_x"

    def forecast(self, train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        train, future = add_price_rel(train, future)
        cols = ["price_rel", "promo"]
        sf = StatsForecast(
            models=[AutoARIMA(season_length=7, approximation=True, alias="yhat")], freq="D"
        )
        out = sf.forecast(
            df=train[KEYS + ["y"] + cols], h=horizon(future), X_df=future[KEYS + cols]
        )
        return self._finish(out, {"yhat": "yhat"})

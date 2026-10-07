"""Model registry."""

from __future__ import annotations

from fdl.models.base import Forecaster
from fdl.models.gbm import LightGBMForecaster
from fdl.models.statistical import ARIMAForecaster, ETSForecaster, SeasonalNaiveForecaster


def build_models(names: list[str], seed: int = 0, neural_steps: int = 200) -> list[Forecaster]:
    """Instantiate forecasters by name. Neural and foundation models import lazily."""
    out: list[Forecaster] = []
    for n in names:
        if n == "seasonal_naive":
            out.append(SeasonalNaiveForecaster())
        elif n == "ets":
            out.append(ETSForecaster())
        elif n == "arima_x":
            out.append(ARIMAForecaster())
        elif n == "lightgbm":
            out.append(LightGBMForecaster(seed=seed))
        elif n in {"nhits", "tft", "patchtst"}:
            from fdl.models.neural import NeuralForecaster

            out.append(NeuralForecaster(n, max_steps=neural_steps, seed=seed))
        elif n == "chronos_bolt":
            from fdl.models.foundation import ChronosForecaster

            out.append(ChronosForecaster())
        else:
            raise ValueError(f"unknown model: {n}")
    return out

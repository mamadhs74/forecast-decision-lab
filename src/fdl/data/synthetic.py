"""Synthetic retail demand with known ground truth.

Why synthetic: the true price elasticity of every series is known, so causal estimators can be
checked against the truth, and injected anomalies are labelled. Real data (see ``m5.py``) is meant for
the forecasting benchmark, where no ground truth is needed.

Demand model (per series i, day t)::

    mu_it = rate_i * dow_i(t) * season_i(t) * trend_i(t) * exp(z_it)
            * (price_it / base_price_i) ** eps_i * (1 + promo_lift * promo_it)
    y_it  ~ NegBin(mean=mu_it, dispersion=k)

``z`` is a persistent latent demand shock (AR(1)).

Price policies
--------------
``confounded``: the price follows a rule on *observed* recent sales (a 7-day mean of ``log(1+y)``
relative to the series' base level), as a retailer repricing on sales would. Sales and price are then
endogenous, and a regression of sales on price without controls is biased towards zero. The rule only
uses lagged observed sales, so controlling for them restores identification (conditional
ignorability holds by construction). ``hidden_confounding`` > 0 additionally lets the price react to
the unobserved lagged shock ``z``, which breaks that assumption on purpose, to show how the
estimators degrade.

``randomized``: price multipliers are drawn independently of everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

PROMO_PRICE_CUT = 0.85
PROMO_LIFT = 0.15
NEGBIN_K = 8.0
ROLL = 7


@dataclass(frozen=True)
class SimConfig:
    n_series: int = 30
    n_days: int = 1095
    start: str = "2022-01-01"
    seed: int = 0
    price_policy: Literal["confounded", "randomized"] = "confounded"
    confounding: float = 0.3  # reaction of log price to recent observed sales
    hidden_confounding: float = 0.0  # reaction of log price to the unobserved lagged shock
    intermittent_share: float = 0.2  # share of low-volume series with many zero days
    min_rate_high: float = 15.0  # lower bound of the daily rate of high-volume series
    anomaly_rate: float = 0.005
    shock_phi: float = 0.9
    shock_sigma: float = 0.12


def simulate_demand(cfg: SimConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(panel, truth)``.

    panel: unique_id, ds, y, price, promo, is_anomaly
    truth: unique_id, elasticity, base_price, base_rate
    """
    rng = np.random.default_rng(cfg.seed)
    n, T = cfg.n_series, cfg.n_days
    dates = pd.date_range(cfg.start, periods=T, freq="D")
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()
    month = dates.month.to_numpy()

    n_low = int(round(cfg.intermittent_share * n))
    rate = np.where(
        np.arange(n) < n_low,
        rng.uniform(0.8, 2.5, n),
        np.clip(rng.lognormal(np.log(35.0), 0.5, n), cfg.min_rate_high, None),
    )
    elasticity = -rng.uniform(1.2, 2.2, n)
    base_price = rng.uniform(2.0, 20.0, n)

    dow_base = np.array([0.90, 0.85, 0.90, 0.95, 1.10, 1.30, 1.00])
    dow_f = (dow_base[None, :] * rng.uniform(0.9, 1.1, (n, 7)))[:, dow]
    phase = rng.uniform(0, 2 * np.pi, (n, 1))
    season = 1 + 0.25 * np.sin(2 * np.pi * doy[None, :] / 365.25 + phase)
    trend = np.exp(rng.normal(0.0, 0.2, (n, 1)) * np.arange(T)[None, :] / T)

    z = np.zeros((n, T))
    shocks = rng.normal(0.0, cfg.shock_sigma, (n, T))
    for t in range(1, T):
        z[:, t] = cfg.shock_phi * z[:, t - 1] + shocks[:, t]

    p_promo = (
        np.where(np.isin(month, [11, 12]), 0.12, 0.07) if cfg.price_policy == "confounded" else 0.08
    )
    promo = (rng.random((n, T)) < p_promo).astype(int)
    price_noise = rng.normal(0.0, 0.05, (n, T))
    random_arm = rng.choice([-0.10, 0.0, 0.10], size=(n, T))
    is_anom = rng.random((n, T)) < cfg.anomaly_rate
    anom_factor = np.where(rng.random((n, T)) < 0.5, 3.0, 0.1)

    ly = np.zeros((n, T))
    y = np.zeros((n, T))
    price = np.zeros((n, T))
    ref = np.log1p(rate)
    for t in range(T):
        if cfg.price_policy == "confounded":
            signal = ly[:, max(0, t - ROLL) : t].mean(axis=1) - ref if t > 0 else np.zeros(n)
            z_prev = z[:, t - 1] if t > 0 else np.zeros(n)
            log_ratio = (
                cfg.confounding * signal + cfg.hidden_confounding * z_prev + price_noise[:, t]
            )
        else:
            log_ratio = random_arm[:, t]
        log_ratio = log_ratio + promo[:, t] * np.log(PROMO_PRICE_CUT)
        price[:, t] = base_price * np.exp(log_ratio)

        mu = (
            rate
            * dow_f[:, t]
            * season[:, t]
            * trend[:, t]
            * np.exp(z[:, t])
            * np.exp(elasticity * log_ratio)
            * (1 + PROMO_LIFT * promo[:, t])
        )
        mu = np.where(is_anom[:, t], mu * anom_factor[:, t], mu)
        lam = rng.gamma(shape=NEGBIN_K, scale=mu / NEGBIN_K)
        y[:, t] = rng.poisson(lam)
        ly[:, t] = np.log1p(y[:, t])

    ids = [f"sku_{i:03d}" for i in range(n)]
    panel = pd.DataFrame(
        {
            "unique_id": np.repeat(ids, T),
            "ds": np.tile(dates, n),
            "y": y.ravel(),
            "price": price.ravel(),
            "promo": promo.ravel(),
            "is_anomaly": is_anom.ravel().astype(int),
        }
    )
    truth = pd.DataFrame(
        {"unique_id": ids, "elasticity": elasticity, "base_price": base_price, "base_rate": rate}
    )
    return panel, truth

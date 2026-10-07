import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from fdl.config import load_config
from fdl.data.m5 import load_m5
from fdl.data.schema import validate_panel
from fdl.data.synthetic import SimConfig, simulate_demand


def test_simulation_is_reproducible():
    a, _ = simulate_demand(SimConfig(n_series=3, n_days=60, seed=5))
    b, _ = simulate_demand(SimConfig(n_series=3, n_days=60, seed=5))
    c, _ = simulate_demand(SimConfig(n_series=3, n_days=60, seed=6))
    pd.testing.assert_frame_equal(a, b)
    assert not a["y"].equals(c["y"])


def test_simulation_shapes_and_truth(small_sim):
    panel, truth = small_sim
    assert panel["unique_id"].nunique() == 6 and len(panel) == 6 * 420
    assert (panel["y"] >= 0).all() and (panel["price"] > 0).all()
    assert (truth["elasticity"] < -1).all()
    assert 0.0 < panel["is_anomaly"].mean() < 0.06


def test_price_policies_differ_in_endogeneity():
    """Confounded prices follow lagged sales; randomized prices must not."""

    def corr(policy: str) -> float:
        p, _ = simulate_demand(
            SimConfig(n_series=20, n_days=500, seed=3, price_policy=policy, intermittent_share=0.0)
        )
        p = p[p["promo"] == 0].copy()
        p["lp"] = np.log(p["price"] / p.groupby("unique_id")["price"].transform("median"))
        p["lag"] = p.groupby("unique_id")["y"].transform(lambda s: np.log1p(s.shift(1)))
        p["lag"] -= p.groupby("unique_id")["lag"].transform("mean")
        return float(p[["lp", "lag"]].corr().iloc[0, 1])

    assert corr("confounded") > 0.2
    assert abs(corr("randomized")) < 0.08


def test_validate_panel_rejects_bad_data(small_panel):
    bad = small_panel.copy()
    bad.loc[0, "y"] = -1.0
    with pytest.raises(pa.errors.SchemaError):
        validate_panel(bad)
    dup = pd.concat([small_panel, small_panel.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        validate_panel(dup)
    gap = small_panel.drop(index=small_panel.index[10])
    with pytest.raises(ValueError, match="missing days"):
        validate_panel(gap)


def test_load_config_validation(tmp_path):
    ok = tmp_path / "ok.yaml"
    ok.write_text("horizon: 7\nn_folds: 4\nn_calib_folds: 2\nmodels: [ets]\ndata: {n_series: 3}\n")
    assert load_config(ok)["horizon"] == 7
    bad = tmp_path / "bad.yaml"
    bad.write_text("horizon: 7\nn_folds: 2\nn_calib_folds: 2\nmodels: [ets]\ndata: {}\n")
    with pytest.raises(ValueError, match="test fold"):
        load_config(bad)
    missing = tmp_path / "missing.yaml"
    missing.write_text("horizon: 7\n")
    with pytest.raises(ValueError, match="missing keys"):
        load_config(missing)


def test_load_m5_on_miniature_files(tmp_path):
    days = 70
    d_cols = [f"d_{i}" for i in range(1, days + 1)]
    rng = np.random.default_rng(0)
    rows = []
    for item in ("FOODS_1_001", "FOODS_1_002"):
        rows.append(
            {
                "id": f"{item}_CA_1_evaluation",
                "item_id": item,
                "dept_id": "FOODS_1",
                "cat_id": "FOODS",
                "store_id": "CA_1",
                "state_id": "CA",
                **dict(zip(d_cols, rng.poisson(5, days), strict=True)),
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "sales_train_evaluation.csv", index=False)
    dates = pd.date_range("2016-01-01", periods=days)
    pd.DataFrame({"d": d_cols, "date": dates, "wm_yr_wk": 11601 + np.arange(days) // 7}).to_csv(
        tmp_path / "calendar.csv", index=False
    )
    weeks = sorted(set(11601 + np.arange(days) // 7))
    prices = []
    for item in ("FOODS_1_001", "FOODS_1_002"):
        for w in weeks:
            p = 4.0
            if item == "FOODS_1_001" and w == weeks[8]:
                p = 3.0  # a 25% price cut after eight stable weeks -> promo proxy
            prices.append({"store_id": "CA_1", "item_id": item, "wm_yr_wk": w, "sell_price": p})
    pd.DataFrame(prices).to_csv(tmp_path / "sell_prices.csv", index=False)

    panel = load_m5(tmp_path, n_series=5)
    assert set(panel["unique_id"]) == {"FOODS_1_001_CA_1", "FOODS_1_002_CA_1"}
    cut_week = panel[(panel["unique_id"] == "FOODS_1_001_CA_1") & (panel["price"] == 3.0)]
    assert len(cut_week) == 7 and (cut_week["promo"] == 1).all()
    assert panel[panel["unique_id"] == "FOODS_1_002_CA_1"]["promo"].sum() == 0

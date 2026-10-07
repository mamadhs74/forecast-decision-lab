import pytest

from fdl.data.schema import validate_panel
from fdl.data.synthetic import SimConfig, simulate_demand


@pytest.fixture(scope="session")
def small_sim():
    panel, truth = simulate_demand(SimConfig(n_series=6, n_days=420, seed=1, anomaly_rate=0.02))
    return validate_panel(panel), truth


@pytest.fixture(scope="session")
def small_panel(small_sim):
    return small_sim[0]

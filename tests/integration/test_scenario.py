"""Phase 2 parity tests: Scenario reproduces OutputAnalysis + TotalMetrics.

The frozen golden snapshots (``tests/golden/data``) are the "no behavior change"
contract. Because Hercules stores data as float32, agreement is asserted at
float32-appropriate tolerance (``rtol=1e-6``); the differences are sub-float32-
epsilon and come only from summation order, not from a change in behavior.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from herc_analysis import channels
from herc_analysis.scenario import Scenario

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "data"
REPO = Path(__file__).resolve().parents[2]
STORED = REPO / "stored_hercules_output"
TESTS = Path(__file__).resolve().parents[1]

# Import the deterministic H5 builder used to generate the golden snapshots, so
# the fixture Scenario is byte-for-byte the same run the golden was frozen from
# (independent of any ambient cwd "hercules_output.h5").
sys.path.insert(0, str(TESTS))
from conftest import _create_test_h5  # noqa: E402


@pytest.fixture(scope="module")
def fixture_h5(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("golden_fixture") / "hercules_output.h5"
    _create_test_h5(str(path))
    return str(path)


# Datasets keyed by golden name -> H5 path. The fixture H5 ("hercules_output.h5")
# is created in the cwd by the session-scoped conftest fixture.
REAL_PATHS = {
    "stored_05": STORED
    / "05_wind_and_storage_with_lmp"
    / "outputs"
    / "hercules_output.h5",
    "stored_02b": STORED
    / "02b_wind_farm_realistic_inflow_precom_floris"
    / "outputs"
    / "hercules_output.h5",
}

RTOL = 1e-6


def _assert_metrics_equal(actual: dict, expected: dict, path: str = "") -> None:
    """Recursively assert two nested metric dicts match (keys + numeric values)."""
    assert set(actual) == set(expected), (
        f"key mismatch at '{path or '<root>'}': "
        f"only-actual={set(actual) - set(expected)}, "
        f"only-expected={set(expected) - set(actual)}"
    )
    for key in expected:
        a, e = actual[key], expected[key]
        here = f"{path}.{key}" if path else key
        if isinstance(e, dict):
            _assert_metrics_equal(a, e, here)
        elif isinstance(e, (int, float)) and not isinstance(e, bool):
            af, ef = float(a), float(e)
            if math.isnan(ef):
                assert math.isnan(af), f"{here}: expected NaN, got {af}"
            else:
                assert af == pytest.approx(ef, rel=RTOL, abs=1e-9), (
                    f"{here}: {af} != {ef}"
                )
        else:
            assert a == e, f"{here}: {a!r} != {e!r}"


def _scenario_for(name: str, fixture_h5: str) -> Scenario:
    if name == "fixture":
        return Scenario(fixture_h5)
    return Scenario(str(REAL_PATHS[name]))


ALL_DATASETS = ("fixture", "stored_05", "stored_02b")


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_metrics_match_golden(name, fixture_h5):
    expected = json.loads((GOLDEN / f"{name}.metrics.json").read_text())
    _assert_metrics_equal(_scenario_for(name, fixture_h5).metrics, expected)


@pytest.mark.parametrize("name", ALL_DATASETS)
def test_monthly_metrics_match_golden(name, fixture_h5):
    path = GOLDEN / f"{name}.monthly.json"
    if not path.exists():
        pytest.skip(f"no golden monthly for {name}")
    # Golden monthly is wrapped as {"monthly": {month: ...}} (the single-scenario
    # return shape of TotalMetrics.compute_monthly_metrics).
    expected = json.loads(path.read_text())["monthly"]
    _assert_metrics_equal(_scenario_for(name, fixture_h5).monthly_metrics, expected)


def test_known_capacity_factor_fixture(fixture_h5):
    # Fixture: 18 MW plant power, 22 MW interconnect -> CF = 18 / 22.
    s = Scenario(fixture_h5)
    assert s.metrics["plant"]["capacity_factor"] == pytest.approx(18 / 22, rel=RTOL)


def test_channels_does_not_copy_raw_and_uses_new_convention(fixture_h5):
    s = Scenario(fixture_h5)
    # New {component}__{signal} naming, power kept in kW.
    assert "battery__power_kw" in s.channels.columns
    # Derived frame is separate from the raw output frame.
    assert s.output.df is not s.channels


def test_channels_physical_equivalence_to_golden(fixture_h5):
    """The new channels carry the same physical quantities as the old df."""
    s = Scenario(fixture_h5)
    gold = pd.read_parquet(GOLDEN / "fixture.channels.parquet")
    new = s.channels
    assert len(new) == len(gold)

    # Per-component: power (kW->MW), energy, RT revenue.
    for comp in s.components:
        n = comp.name
        np.testing.assert_allclose(
            channels.kw_to_mw(new[f"{n}__power_kw"]).to_numpy(),
            gold[f"{n}_power_mw"].to_numpy(),
            rtol=RTOL,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            new[f"{n}__energy_mwh"].to_numpy(),
            gold[f"{n}_energy_mwh"].to_numpy(),
            rtol=RTOL,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            new[f"{n}__revenue_rt_usd"].to_numpy(),
            gold[f"{n}_revenue_rt"].to_numpy(),
            rtol=RTOL,
            atol=1e-9,
        )

    # Plant-level: total power and surplus capacity headroom.
    np.testing.assert_allclose(
        new["plant__total_power_mw"].to_numpy(),
        gold["total_plant_power_mw"].to_numpy(),
        rtol=RTOL,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        new["plant__surplus_capacity_mw"].to_numpy(),
        gold["surplus_capacity_mw"].to_numpy(),
        rtol=RTOL,
        atol=1e-9,
    )


def test_construction_is_cheap(fixture_h5):
    """Constructing a Scenario discovers components without building channels."""
    s = Scenario(fixture_h5)
    assert "channels" not in s.__dict__  # cached_property not yet computed
    assert [c.name for c in s.components]  # discovery happened
    _ = s.channels
    assert "channels" in s.__dict__  # now cached


def test_metric_set_default_resolutions_exclude_monthly(fixture_h5):
    s = Scenario(fixture_h5)  # default ("total", "annual")
    resolutions = set(s.metric_set.rows["resolution"])
    assert resolutions == {"total", "annual"}
    assert "monthly" not in resolutions


def test_metric_set_includes_monthly_only_when_requested(fixture_h5):
    s = Scenario(fixture_h5, resolutions=("total", "monthly"))
    resolutions = set(s.metric_set.rows["resolution"])
    assert resolutions == {"total", "monthly"}
    # monthly rows reproduce the nested monthly_metrics values.
    assert s.metric_set.at("monthly")["period"].nunique() == len(s.monthly_metrics)


def test_metrics_at_buckets_by_calendar(fixture_h5):
    s = Scenario(fixture_h5)
    total = s.metrics_at("total")
    assert set(total) == {"total"} and total["total"] is s.metrics
    yearly = s.metrics_at("yearly")
    assert set(yearly) == {"2024"}  # fixture spans Jan 2024


def test_metrics_at_rejects_annual(fixture_h5):
    s = Scenario(fixture_h5)
    with pytest.raises(ValueError, match="averaged annual"):
        s.metrics_at("annual")


def test_annual_resolution_is_total_over_sim_years(fixture_h5):
    s = Scenario(fixture_h5, resolutions=("total", "annual"))
    ms = s.metric_set
    sim_years = s.meta.sim_years
    # extensive: annual == total / sim_years
    total_energy = ms.scalar("plant", "energy_mwh", resolution="total", period="total")
    annual_energy = ms.scalar(
        "plant", "energy_mwh", resolution="annual", period="annual"
    )
    assert annual_energy == pytest.approx(total_energy / sim_years)
    # intensive: annual == total (a rate; unchanged)
    total_cf = ms.scalar("plant", "capacity_factor", resolution="total", period="total")
    annual_cf = ms.scalar(
        "plant", "capacity_factor", resolution="annual", period="annual"
    )
    assert annual_cf == pytest.approx(total_cf)

    # The scaling tag describes the metric's nature, not the resolution: energy
    # is 'extensive' at both 'total' and 'annual'.
    rows = ms.rows
    energy = rows[(rows["entity"] == "plant") & (rows["metric"] == "energy_mwh")]
    assert set(energy["scaling"]) == {"extensive"}


def test_yearly_resolution_buckets_per_year(fixture_h5):
    s = Scenario(fixture_h5, resolutions=("total", "yearly"))
    ms = s.metric_set
    yearly = ms.at("yearly")
    assert set(yearly["period"]) == {"2024"}


def test_unsupported_resolution_rejected(fixture_h5):
    with pytest.raises(ValueError, match="Unsupported resolution"):
        Scenario(fixture_h5, resolutions=("weekly",))

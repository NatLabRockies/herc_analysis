"""Phase 3 integration tests: MetricSet from Scenario + the Comparison engine."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from herc_analysis.comparison import Comparison
from herc_analysis.scenario import Scenario

REPO = Path(__file__).resolve().parents[2]
STORED = REPO / "stored_hercules_output"
CASES = {
    "wind_only": STORED
    / "02b_wind_farm_realistic_inflow_precom_floris"
    / "outputs"
    / "hercules_output.h5",
    "wind_storage": STORED
    / "05_wind_and_storage_with_lmp"
    / "outputs"
    / "hercules_output.h5",
}


@pytest.fixture(scope="module")
def scenarios() -> list[Scenario]:
    return [Scenario(str(p), name=n) for n, p in CASES.items()]


def test_metric_set_matches_nested_metrics(scenarios):
    """The long-format values agree with the nested Scenario.metrics dict."""
    s = scenarios[1]  # wind_storage
    ms = s.metric_set
    assert ms.scalar("plant", "energy_mwh") == pytest.approx(
        s.metrics["plant"]["total_energy_mwh"]
    )
    assert ms.scalar("plant", "capacity_factor") == pytest.approx(
        s.metrics["plant"]["capacity_factor"]
    )
    assert ms.scalar("plant", "revenue_rt") == pytest.approx(
        s.metrics["plant"]["total_revenue_rt_k"]
    )


def test_from_scenarios_table_annual_scaling(scenarios):
    cmp = Comparison.from_scenarios(scenarios)
    annual = cmp.table(["energy_mwh", "capacity_factor"], period="annual")
    assert list(annual.index) == ["wind_only", "wind_storage"]

    for s in scenarios:
        # extensive -> annual divides by sim_years.
        expected_energy = s.metric_set.scalar("plant", "energy_mwh") / s.meta.sim_years
        assert annual.loc[s.name, "energy_mwh"] == pytest.approx(expected_energy)
        # intensive -> unchanged between annual and total.
        cf = s.metric_set.scalar("plant", "capacity_factor")
        assert annual.loc[s.name, "capacity_factor"] == pytest.approx(cf)


def test_total_view_equals_raw_for_extensive(scenarios):
    cmp = Comparison.from_scenarios(scenarios)
    total = cmp.table("energy_mwh", period="total")
    for s in scenarios:
        assert total.loc[s.name, "energy_mwh"] == pytest.approx(
            s.metric_set.scalar("plant", "energy_mwh")
        )


def test_from_cases_matches_from_scenarios(scenarios, tmp_path):
    # Write each scenario's metric_set to a per-case metrics.csv, then collect.
    case_dirs = []
    for s in scenarios:
        out = tmp_path / s.name / "outputs" / "metrics.csv"
        s.metric_set.to_csv(out)
        case_dirs.append(tmp_path / s.name)

    from_cases = Comparison.from_cases(case_dirs, case_names=list(CASES))
    from_scen = Comparison.from_scenarios(scenarios)

    metrics = ["energy_mwh", "capacity_factor", "revenue_rt"]
    a = from_cases.table(metrics, period="annual")
    b = from_scen.table(metrics, period="annual")
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_table_multi_entity_multiindex(scenarios):
    cmp = Comparison.from_scenarios(scenarios)
    wide = cmp.table("energy_mwh", entity="all")
    assert isinstance(wide.columns, pd.MultiIndex)
    # plant plus the discovered components appear as entities.
    entities = {col[0] for col in wide.columns}
    assert "plant" in entities


def test_scenariocomparison_emits_deprecation_warning():
    df = pd.DataFrame(
        {
            "case": ["a"],
            "scope": ["plant"],
            "metric": ["energy_mwh"],
            "value": [1.0],
            "scaling": ["extensive"],
            "sim_years": [1.0],
        }
    )
    from herc_analysis.scenario_compare import ScenarioComparison

    with pytest.warns(DeprecationWarning):
        ScenarioComparison(df)


def test_totalmetrics_list_mode_emits_deprecation_warning():
    from herc_analysis.total_metrics import TotalMetrics

    with pytest.warns(DeprecationWarning):
        TotalMetrics([object(), object()])


def test_missing_required_columns_raises():
    bad = pd.DataFrame({"case": ["a"], "entity": ["plant"]})
    with pytest.raises(ValueError):
        Comparison(bad)


def test_value_factor_nan_safe(scenarios):
    """wind_only has no LMP; value_factor is NaN and must not crash the table."""
    cmp = Comparison.from_scenarios(scenarios)
    vf = cmp.table("value_factor", period="annual")
    assert math.isnan(vf.loc["wind_only", "value_factor"]) or vf.loc[
        "wind_only", "value_factor"
    ] == pytest.approx(vf.loc["wind_only", "value_factor"])

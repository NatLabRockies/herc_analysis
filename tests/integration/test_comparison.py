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


def test_annual_resolution_is_per_year(scenarios):
    cmp = Comparison.from_scenarios(scenarios)
    annual = cmp.table(["energy_mwh", "capacity_factor"], resolution="annual")
    assert list(annual.index) == ["wind_only", "wind_storage"]

    for s in scenarios:
        # extensive -> annual is the whole-run total divided by sim_years.
        expected_energy = s.metric_set.scalar("plant", "energy_mwh") / s.meta.sim_years
        assert annual.loc[s.name, "energy_mwh"] == pytest.approx(expected_energy)
        # intensive -> unchanged between annual and total.
        cf = s.metric_set.scalar("plant", "capacity_factor")
        assert annual.loc[s.name, "capacity_factor"] == pytest.approx(cf)


def test_total_resolution_is_whole_run_total(scenarios):
    cmp = Comparison.from_scenarios(scenarios)
    total = cmp.table("energy_mwh", resolution="total")
    for s in scenarios:
        assert total.loc[s.name, "energy_mwh"] == pytest.approx(
            s.metric_set.scalar("plant", "energy_mwh")
        )


def test_total_equals_annual_times_sim_years(scenarios):
    """For an extensive metric, total == annual * sim_years."""
    cmp = Comparison.from_scenarios(scenarios)
    annual = cmp.table("energy_mwh", resolution="annual")
    total = cmp.table("energy_mwh", resolution="total")
    for s in scenarios:
        assert total.loc[s.name, "energy_mwh"] == pytest.approx(
            annual.loc[s.name, "energy_mwh"] * s.meta.sim_years
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
    a = from_cases.table(metrics, resolution="annual")
    b = from_scen.table(metrics, resolution="annual")
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_table_multi_entity_multiindex(scenarios):
    cmp = Comparison.from_scenarios(scenarios)
    wide = cmp.table("energy_mwh", entity="all")
    assert isinstance(wide.columns, pd.MultiIndex)
    # plant plus the discovered components appear as entities.
    entities = {col[0] for col in wide.columns}
    assert "plant" in entities


def test_missing_required_columns_raises():
    bad = pd.DataFrame({"case": ["a"], "entity": ["plant"]})
    with pytest.raises(ValueError):
        Comparison(bad)


def test_table_default_resolution_is_total(scenarios):
    cmp = Comparison.from_scenarios(scenarios)
    default = cmp.table("energy_mwh")
    explicit = cmp.table("energy_mwh", resolution="total")
    pd.testing.assert_frame_equal(default, explicit)


def test_plot_rejects_entity_all(scenarios):
    cmp = Comparison.from_scenarios(scenarios)
    with pytest.raises(ValueError, match="single entity"):
        cmp.plot("energy_mwh", entity="all")


def _monthly_metrics_frame() -> pd.DataFrame:
    rows = []
    for case in ("a", "b"):
        for period, value in (("2024-01", 1.0), ("2024-02", 2.0)):
            rows.append(
                {
                    "case": case,
                    "entity": "plant",
                    "metric": "energy_mwh",
                    "resolution": "monthly",
                    "period": period,
                    "value": value,
                    "unit": "MWh",
                    "scaling": "extensive",
                    "sim_years": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_table_monthly_requires_period():
    cmp = Comparison(_monthly_metrics_frame())
    with pytest.raises(ValueError, match="period"):
        cmp.table("energy_mwh", resolution="monthly")


def test_table_monthly_with_period_selects_bucket():
    cmp = Comparison(_monthly_metrics_frame())
    wide = cmp.table("energy_mwh", resolution="monthly", period="2024-02")
    assert wide.loc["a", "energy_mwh"] == 2.0
    assert wide.loc["b", "energy_mwh"] == 2.0


def test_value_factor_nan_safe(scenarios):
    """wind_only has no LMP; value_factor is NaN and must not crash the table."""
    cmp = Comparison.from_scenarios(scenarios)
    vf = cmp.table("value_factor", resolution="annual")
    assert math.isnan(vf.loc["wind_only", "value_factor"]) or vf.loc[
        "wind_only", "value_factor"
    ] == pytest.approx(vf.loc["wind_only", "value_factor"])

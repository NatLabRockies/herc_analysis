import numpy as np
import pytest

from herc_analysis import OutputAnalysis, TotalMetrics


def test_total_metrics_single_scenario():
    """Test computing metrics for a single OutputAnalysis object."""
    oa = OutputAnalysis("hercules_output.h5")
    tm = TotalMetrics(oa)
    metrics = tm.compute_metrics(display=False)

    # Plant capacity factor
    np.testing.assert_allclose(metrics["plant"]["capacity_factor"], 18 / 22)

    # Per-component energy
    np.testing.assert_allclose(
        metrics["components"]["wind_farm"]["energy_mwh"],
        3 * 15.0 * 2 / 3600,
    )
    np.testing.assert_allclose(
        metrics["components"]["solar_farm"]["energy_mwh"],
        3 * 1.0 * 2 / 3600,
    )
    np.testing.assert_allclose(
        metrics["components"]["battery"]["energy_mwh"],
        3 * 2.0 * 2 / 3600,
    )

    # Plant total energy
    np.testing.assert_allclose(
        metrics["plant"]["total_energy_mwh"],
        3 * 18.0 * 2 / 3600,
    )

    # Revenue
    np.testing.assert_allclose(
        metrics["plant"]["total_revenue_rt_k"],
        3 * 18.0 * 2 / 3600 * 10.0 / 1e3,
    )

    # Surplus
    np.testing.assert_allclose(
        metrics["plant"]["surplus_capacity_revenue_rt_k"],
        3 * 4.0 * 2 / 3600 * 10.0 / 1e3,
    )


def test_category_aggregates():
    """Test category-level aggregate metrics."""
    oa = OutputAnalysis("hercules_output.h5")
    tm = TotalMetrics(oa)
    metrics = tm.compute_metrics(display=False)

    gen = metrics["categories"]["generator"]
    np.testing.assert_allclose(
        gen["energy_mwh"],
        3 * (15.0 + 1.0) * 2 / 3600,
    )

    stor = metrics["categories"]["storage"]
    np.testing.assert_allclose(
        stor["energy_mwh"],
        3 * 2.0 * 2 / 3600,
    )


def test_storage_metrics():
    """Test storage-specific discharge/charge metrics."""
    oa = OutputAnalysis("hercules_output.h5")
    tm = TotalMetrics(oa)
    metrics = tm.compute_metrics(display=False)

    batt = metrics["components"]["battery"]
    assert "energy_discharge_mwh" in batt
    assert "energy_charge_mwh" in batt
    assert "revenue_rt_discharge_k" in batt
    assert "revenue_rt_charge_k" in batt


def test_battery_mileage_metrics_present_and_correct():
    """Battery mileage equals the summed absolute SOC step differences."""
    oa = OutputAnalysis("hercules_output.h5")
    tm = TotalMetrics(oa)
    metrics = tm.compute_metrics(display=False)

    batt = metrics["components"]["battery"]
    assert "battery_mileage_soc" in batt
    assert "battery_mileage_mwh" in batt

    expected_soc = oa.df["battery_soc"].diff().abs().sum()
    np.testing.assert_allclose(batt["battery_mileage_soc"], expected_soc)

    energy_cap_mwh = oa.h_dict["battery"]["energy_capacity"] / 1000.0
    np.testing.assert_allclose(
        batt["battery_mileage_mwh"], expected_soc * energy_cap_mwh
    )


def test_battery_mileage_constant_soc_is_zero():
    """A flat SOC trace yields zero mileage."""
    oa = OutputAnalysis("hercules_output.h5")
    tm = TotalMetrics(oa)
    metrics = tm.compute_metrics(display=False)

    # The test fixture holds SOC constant at 1.0, so mileage must be zero.
    np.testing.assert_allclose(
        metrics["components"]["battery"]["battery_mileage_soc"], 0.0
    )


def test_available_metrics():
    """Test that available_metrics returns paths."""
    oa = OutputAnalysis("hercules_output.h5")
    tm = TotalMetrics(oa)
    tm.compute_metrics(display=False)
    paths = tm.available_metrics()
    assert "plant.total_energy_mwh" in paths
    assert "components.wind_farm.energy_mwh" in paths


def test_compare_scenarios_include_pct_change_requires_two_scenarios():
    """include_pct_change raises when there are not exactly two scenarios."""
    oa = OutputAnalysis("hercules_output.h5")
    tm = TotalMetrics(
        [oa, oa, oa],
        scenario_names=["A", "B", "C"],
    )
    tm.compute_metrics(display=False)
    with pytest.raises(ValueError, match="exactly two"):
        tm.compare_scenarios(display_format="raw", include_pct_change=True)


def test_format_pct_change_signed_two_decimals():
    """Percent change uses two decimals and an explicit sign."""
    assert TotalMetrics._format_pct_change(100.0, 124.15) == "+24.15%"
    assert TotalMetrics._format_pct_change(100.0, 75.5) == "-24.50%"


def test_compare_scenarios_include_pct_change_identical_baselines():
    """Two identical scenarios yield +0.00% on comparable numeric rows."""
    oa = OutputAnalysis("hercules_output.h5")
    tm = TotalMetrics([oa, oa], scenario_names=["S1", "S2"])
    tm.compute_metrics(display=False)
    df = tm.compare_scenarios(display_format="raw", include_pct_change=True)
    col = TotalMetrics._PCT_CHANGE_COLUMN
    assert col in df.columns
    assert df.loc["  Plant Energy (MWh)", col] == "+0.00%"

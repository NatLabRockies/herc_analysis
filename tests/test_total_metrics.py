import numpy as np

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


def test_available_metrics():
    """Test that available_metrics returns paths."""
    oa = OutputAnalysis("hercules_output.h5")
    tm = TotalMetrics(oa)
    tm.compute_metrics(display=False)
    paths = tm.available_metrics()
    assert "plant.total_energy_mwh" in paths
    assert "components.wind_farm.energy_mwh" in paths

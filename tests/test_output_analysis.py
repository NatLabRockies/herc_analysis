import numpy as np

from herc_analysis import OutputAnalysis


def test_output_analysis_basic_init():
    """Test that OutputAnalysis loads h5 file successfully."""
    oa = OutputAnalysis("hercules_output.h5")
    assert oa.h_dict is not None


def test_component_discovery():
    """Test generic component discovery via component_type."""
    oa = OutputAnalysis("hercules_output.h5")
    names = [c.name for c in oa.components]
    assert "wind_farm" in names
    assert "solar_farm" in names
    assert "battery" in names

    assert oa.generators == ["wind_farm", "solar_farm"]
    assert oa.storage == ["battery"]
    assert oa.loads == []


def test_component_info_fields():
    """Test ComponentInfo type and category for discovered components."""
    oa = OutputAnalysis("hercules_output.h5")
    wind = oa.get_component("wind_farm")
    assert wind is not None
    assert wind.component_type == "WindFarm"
    assert wind.category == "generator"

    batt = oa.get_component("battery")
    assert batt is not None
    assert batt.component_type == "BatterySimple"
    assert batt.category == "storage"


def test_per_component_derived_columns():
    """Test that per-component power, energy, revenue columns are computed."""
    oa = OutputAnalysis("hercules_output.h5")

    assert oa.df["wind_farm_power_mw"].tolist() == [15.0] * 3
    assert oa.df["solar_farm_power_mw"].tolist() == [1.0] * 3
    assert oa.df["battery_power_mw"].tolist() == [2.0] * 3

    assert oa.df["wind_farm_energy_mwh"].tolist() == [15.0 * 2 / 3600] * 3
    assert oa.df["solar_farm_energy_mwh"].tolist() == [1.0 * 2 / 3600] * 3
    assert oa.df["battery_energy_mwh"].tolist() == [2.0 * 2 / 3600] * 3


def test_storage_specific_columns():
    """Test that storage components get charge/discharge and SOC columns."""
    oa = OutputAnalysis("hercules_output.h5")

    assert "battery_revenue_rt_discharge" in oa.df.columns
    assert "battery_revenue_rt_charge" in oa.df.columns
    assert "battery_soc" in oa.df.columns


def test_category_aggregates():
    """Test category-level aggregate columns."""
    oa = OutputAnalysis("hercules_output.h5")

    np.testing.assert_allclose(
        oa.df["generator_total_power_mw"].values,
        oa.df["wind_farm_power_mw"].values + oa.df["solar_farm_power_mw"].values,
    )
    np.testing.assert_allclose(
        oa.df["storage_total_power_mw"].values,
        oa.df["battery_power_mw"].values,
    )


def test_plant_level_metrics():
    """Test plant-level totals and surplus capacity."""
    oa = OutputAnalysis("hercules_output.h5")

    assert oa.interconnect_mw == 22
    assert oa.df["total_plant_power_mw"].tolist() == [18.0] * 3
    assert oa.df["surplus_capacity_mw"].tolist() == [4.0] * 3

    np.testing.assert_allclose(
        oa.df["total_plant_energy_mwh"].values,
        [18.0 * 2 / 3600] * 3,
    )


def test_revenue_columns():
    """Test revenue derived columns."""
    oa = OutputAnalysis("hercules_output.h5")

    assert oa.df["lmp_rt"].tolist() == [10.0] * 3

    np.testing.assert_allclose(
        oa.df["wind_farm_revenue_rt"].values,
        [15.0 * 2 / 3600 * 10.0] * 3,
    )
    np.testing.assert_allclose(
        oa.df["total_plant_revenue_rt"].values,
        [18.0 * 2 / 3600 * 10.0] * 3,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        oa.df["surplus_capacity_revenue_rt"].values,
        [4.0 * 2 / 3600 * 10.0] * 3,
        atol=1e-6,
    )


def test_get_signal_columns():
    """Test the get_signal_columns helper for array signals."""
    oa = OutputAnalysis("hercules_output.h5")
    tp = oa.get_signal_columns("wind_farm", "turbine_powers")
    assert len(tp) == 2
    assert all("turbine_powers" in c for c in tp)

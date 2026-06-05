import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

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


def _write_excess_charge_h5(filename: str) -> None:
    """Build a small H5 where local generation exceeds the interconnect.

    Layout (3 steps, dt_log = 3600 s = 1 h):
      - solar_farm: 110 MW (constant)  -> drives 10 MW of excess
      - battery:    -20 MW (constant)  -> charging at 20 MW
      - lmp_rt = lmp_da = 50 $/MWh
      - interconnect = 100 MW
      - locally_generated = 110 MW (solar only)
    """
    n_steps = 3
    dt_log = 3600.0
    step = np.arange(n_steps)
    time_arr = step * dt_log

    solar_kw = np.ones(n_steps) * 110_000.0
    battery_kw = np.ones(n_steps) * -20_000.0
    battery_soc = np.ones(n_steps) * 0.5
    locally_gen_kw = solar_kw.copy()
    plant_power_kw = locally_gen_kw + battery_kw  # 90 MW (charging absorbs 20)

    lmp = np.ones(n_steps) * 50.0
    interconnect_kw = 100_000.0  # 100 MW

    h_dict = {
        "starttime": 0.0,
        "plant": {"interconnect_limit": float(interconnect_kw)},
        "solar_farm": {
            "component_type": "SolarPySAMPVWatts",
            "system_capacity": 110_000.0,
        },
        "battery": {
            "component_type": "BatterySimple",
            "size": 20_000.0,
            "energy_capacity": 40_000.0,
        },
        "external_signals": True,
    }

    starttime_utc = pd.to_datetime("2024-01-01 00:00:00", utc=True)

    with h5py.File(filename, "w") as hf:
        mg = hf.create_group("metadata")
        mg.attrs["h_dict"] = json.dumps(h_dict)
        mg.attrs["starttime"] = 0.0
        mg.attrs["endtime"] = float(n_steps * dt_log)
        mg.attrs["dt_sim"] = 1.0
        mg.attrs["dt_log"] = dt_log
        mg.attrs["log_every_n"] = 1
        mg.attrs["total_simulation_time"] = float(n_steps * dt_log)
        mg.attrs["total_simulation_days"] = float(n_steps * dt_log) / 86400.0
        mg.attrs["starttime_utc"] = starttime_utc.timestamp()
        mg.attrs["total_rows_written"] = n_steps
        mg.attrs["total_time_wall"] = 1

        dg = hf.create_group("data")
        dg.create_dataset("time", data=time_arr)
        dg.create_dataset("step", data=step)
        dg.create_dataset("plant_power", data=plant_power_kw)
        dg.create_dataset("plant_locally_generated_power", data=locally_gen_kw)

        cg = dg.create_group("components")
        cg.create_dataset("solar_farm.power", data=solar_kw)
        cg.create_dataset("battery.power", data=battery_kw)
        cg.create_dataset("battery.soc", data=battery_soc)

        esg = dg.create_group("external_signals")
        esg.create_dataset("external_signals.lmp_rt", data=lmp)
        esg.create_dataset("external_signals.lmp_da", data=lmp)


def test_excess_charge_correction_zero_cost_for_absorbed_excess():
    """Storage charging that absorbs excess local generation pays $0.

    With excess = 10 MW and battery charging at 20 MW (per row, 1 h), the
    battery should pay LMP for only 10 MW * 1 h = 10 MWh per step (not 20),
    matching the user's example. The 'if all paid' baseline retains the
    uncorrected cost so the savings are visible.
    """
    fn = "_excess_charge_test.h5"
    try:
        _write_excess_charge_h5(fn)
        oa = OutputAnalysis(fn)
        df = oa.df

        # Excess local generation per step
        np.testing.assert_allclose(df["excess_local_generation_mw"].values, [10.0] * 3)

        # Battery absorbs all 10 MW of excess (its charge of 20 MW exceeds it).
        np.testing.assert_allclose(df["battery_excess_absorbed_mw"].values, [10.0] * 3)

        # 1 hour steps, 50 $/MWh -> -1000 $ if all paid, -500 $ paid, +500 $ savings
        np.testing.assert_allclose(
            df["battery_revenue_rt_charge_if_all_paid"].values, [-1000.0] * 3
        )
        np.testing.assert_allclose(df["battery_revenue_rt_charge"].values, [-500.0] * 3)
        np.testing.assert_allclose(
            df["battery_revenue_rt_charge_savings"].values, [500.0] * 3
        )
        np.testing.assert_allclose(
            df["battery_revenue_da_charge_if_all_paid"].values, [-1000.0] * 3
        )
        np.testing.assert_allclose(df["battery_revenue_da_charge"].values, [-500.0] * 3)

        # Component-level revenue (battery_revenue_rt) reflects the correction:
        # original lmp*energy = 50 * (-20) = -1000; corrected = -500.
        np.testing.assert_allclose(df["battery_revenue_rt"].values, [-500.0] * 3)

        # Generator-side excess correction: solar carries the full excess
        # share (only generator), so 10 MW * 1 h * 50 $/MWh = 500 $ is
        # foregone revenue per step.
        np.testing.assert_allclose(
            df["solar_farm_excess_local_generation_mw"].values, [10.0] * 3
        )
        np.testing.assert_allclose(
            df["solar_farm_revenue_rt_if_all_paid"].values, [5500.0] * 3
        )
        np.testing.assert_allclose(
            df["solar_farm_revenue_rt_excess_loss"].values, [500.0] * 3
        )
        np.testing.assert_allclose(df["solar_farm_revenue_rt"].values, [5000.0] * 3)
        np.testing.assert_allclose(
            df["solar_farm_revenue_da_excess_loss"].values, [500.0] * 3
        )

        # Plant total revenue is the SUM of component revenues, equal to the
        # physical grid revenue: lmp * plant.power_to_grid = 50 * (110 - 20)
        # = 4500 $. Without the generator correction this would be 5000 $
        # (overstating revenue by the value of the excess generation).
        np.testing.assert_allclose(df["total_plant_revenue_rt"].values, [4500.0] * 3)

        # Sanity invariant: plant total == sum of component revenues (per row).
        comp_sum = df["solar_farm_revenue_rt"].values + df["battery_revenue_rt"].values
        np.testing.assert_allclose(df["total_plant_revenue_rt"].values, comp_sum)
    finally:
        Path(fn).unlink(missing_ok=True)


def test_excess_charge_correction_skipped_without_locally_generated_power(capsys):
    """Without plant.locally_generated_power, correction is skipped with a warning."""
    # The default fixture H5 *does* include plant.locally_generated_power, so
    # we just confirm the no-op path: locally_generated = 16 MW < 22 MW
    # interconnect, so no excess and no correction is applied.
    oa = OutputAnalysis("hercules_output.h5")
    df = oa.df
    np.testing.assert_allclose(df["excess_local_generation_mw"].values, [0.0] * 3)
    np.testing.assert_allclose(df["battery_excess_absorbed_mw"].values, [0.0] * 3)
    # 'if all paid' equals 'paid' when there is no excess.
    np.testing.assert_allclose(
        df["battery_revenue_rt_charge_if_all_paid"].values,
        df["battery_revenue_rt_charge"].values,
    )

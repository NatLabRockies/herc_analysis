"""Shared test fixtures for herc_analysis tests.

Generates a small, deterministic Hercules H5 file with known values so that
tests can assert exact numeric results.

Layout (testing_values=True, 3 time steps, dt_log=2 s):
    - wind_farm  (WindFarm):        2 turbines, powers=[5000, 10000] kW -> 15 MW
    - solar_farm (SolarPySAMPVWatts): power=1000 kW -> 1 MW
    - battery    (BatterySimple):     power=2000 kW -> 2 MW, soc=1.0
    - lmp_rt = lmp_da = 10 $/MWh
    - interconnect = 22 MW
    - total plant power = 18 MW
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

TEST_H5 = "hercules_output.h5"
DEMO_TEST_H5 = "demo_test.h5"

N_STEPS = 3
DT_SIM = 1.0
DT_LOG = 2.0


def _create_test_h5(filename: str) -> None:
    """Write a minimal, deterministic Hercules H5 file for testing."""
    step_array = np.arange(N_STEPS)
    time_array = step_array * DT_LOG

    starttime = 0.0
    endtime = N_STEPS * DT_LOG
    starttime_utc = pd.to_datetime("2024-01-01 00:00:00", utc=True)

    n_turbines = 2
    rated_turbine_power = 10000.0

    wind_power_mat = (np.ones(N_STEPS) * 5000, np.ones(N_STEPS) * 10000)
    wind_power_total = np.sum(wind_power_mat, axis=0)
    wind_speeds_background = np.ones(N_STEPS)
    wind_speeds_withwakes = np.ones(N_STEPS)
    wind_directions = np.ones(N_STEPS)

    solar_power = np.ones(N_STEPS) * 1000
    solar_system_capacity = 2000.0
    solar_ghi = np.ones(N_STEPS)
    solar_dni = np.ones(N_STEPS)
    solar_dhi = np.ones(N_STEPS)
    solar_poa = np.ones(N_STEPS)

    battery_size = 2000.0
    battery_power = np.ones(N_STEPS) * battery_size
    battery_soc = np.ones(N_STEPS)
    battery_energy_capacity = battery_size * 2

    lmp_rt = np.ones(N_STEPS) * 10.0
    lmp_da = np.ones(N_STEPS) * 10.0

    interconnect_limit = 20000 + 2000  # wind + solar capacity

    plant_power = wind_power_total + solar_power + battery_power
    plant_locally_generated_power = wind_power_total + solar_power

    h_dict = {
        "starttime": starttime,
        "plant": {"interconnect_limit": float(interconnect_limit)},
        "wind_farm": {
            "component_type": "WindFarm",
            "n_turbines": int(n_turbines),
            "rated_turbine_power": float(rated_turbine_power),
        },
        "solar_farm": {
            "component_type": "SolarPySAMPVWatts",
            "system_capacity": float(solar_system_capacity),
        },
        "battery": {
            "component_type": "BatterySimple",
            "size": float(battery_size),
            "energy_capacity": float(battery_energy_capacity),
        },
        "external_signals": True,
    }

    with h5py.File(filename, "w") as hf:
        mg = hf.create_group("metadata")
        mg.attrs["h_dict"] = json.dumps(h_dict)
        mg.attrs["starttime"] = starttime
        mg.attrs["endtime"] = endtime
        mg.attrs["dt_sim"] = DT_SIM
        mg.attrs["dt_log"] = DT_LOG
        mg.attrs["log_every_n"] = 2
        mg.attrs["total_simulation_time"] = endtime - starttime
        mg.attrs["total_simulation_days"] = (endtime - starttime) / 86400.0
        mg.attrs["starttime_utc"] = starttime_utc.timestamp()
        mg.attrs["total_rows_written"] = len(time_array)
        mg.attrs["total_time_wall"] = 10

        dg = hf.create_group("data")
        dg.create_dataset("time", data=time_array)
        dg.create_dataset("step", data=step_array)
        dg.create_dataset("plant_power", data=plant_power)
        dg.create_dataset(
            "plant_locally_generated_power", data=plant_locally_generated_power
        )

        cg = dg.create_group("components")
        cg.create_dataset("wind_farm.power", data=wind_power_total)
        cg.create_dataset(
            "wind_farm.wind_speed_mean_background", data=wind_speeds_background
        )
        cg.create_dataset(
            "wind_farm.wind_speed_mean_withwakes", data=wind_speeds_withwakes
        )
        cg.create_dataset("wind_farm.wind_direction_mean", data=wind_directions)
        for i in range(n_turbines):
            cg.create_dataset(
                f"wind_farm.turbine_powers.{i:03d}", data=wind_power_mat[i]
            )

        cg.create_dataset("solar_farm.power", data=solar_power)
        cg.create_dataset("solar_farm.dni", data=solar_dni)
        cg.create_dataset("solar_farm.poa", data=solar_poa)
        cg.create_dataset("solar_farm.ghi", data=solar_ghi)
        cg.create_dataset("solar_farm.dhi", data=solar_dhi)

        cg.create_dataset("battery.power", data=battery_power)
        cg.create_dataset("battery.soc", data=battery_soc)

        esg = dg.create_group("external_signals")
        esg.create_dataset("external_signals.lmp_rt", data=lmp_rt)
        esg.create_dataset("external_signals.lmp_da", data=lmp_da)


@pytest.fixture(autouse=True, scope="session")
def test_h5_files():
    """Create deterministic test H5 files before any tests run."""
    _create_test_h5(TEST_H5)
    _create_test_h5(DEMO_TEST_H5)
    yield
    for f in (TEST_H5, DEMO_TEST_H5):
        Path(f).unlink(missing_ok=True)

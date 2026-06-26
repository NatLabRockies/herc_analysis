"""Unit tests for the capacity availability providers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from herc_analysis.capacity import (
    BatteryAvailability,
    FixedAvailability,
    FromRunAvailability,
)
from herc_analysis.capacity._miso_engine import compute_battery_availability


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_utc": pd.date_range(
                "2022-09-01 05:00", periods=4, freq="h", tz="UTC"
            ),
            "batt__power_kw": [100.0, -50.0, 0.0, 200.0],
            "batt__soc": [1.0, 0.9, 0.8, 0.85],
        }
    )


def test_fixed_availability_is_constant_copy():
    df = _frame()
    out = FixedAvailability(42.0).attach(df, "batt")
    assert (out["batt__availability"] == 42.0).all()
    assert "batt__availability" not in df.columns  # original untouched


def test_from_run_availability_aligns_by_time():
    df = _frame()
    source = pd.DataFrame(
        {"time_utc": df["time_utc"], "avail": [10.0, 20.0, 30.0, 40.0]}
    )
    out = FromRunAvailability(source, "avail").attach(df, "batt")
    np.testing.assert_allclose(
        out["batt__availability"].to_numpy(), [10.0, 20.0, 30.0, 40.0]
    )


def test_from_run_availability_reads_csv(tmp_path):
    df = _frame()
    source = pd.DataFrame({"time_utc": df["time_utc"], "avail": [1.0, 2.0, 3.0, 4.0]})
    path = tmp_path / "src.csv"
    source.to_csv(path, index=False)
    out = FromRunAvailability(path, "avail").attach(df, "batt")
    np.testing.assert_allclose(
        out["batt__availability"].to_numpy(), [1.0, 2.0, 3.0, 4.0]
    )


def test_battery_availability_matches_compute_function():
    df = _frame()
    provider = BatteryAvailability(
        rated_power_kw=100.0,
        rated_energy_kwh=200.0,
        min_soc=0.1,
        eta_discharge=0.95,
    )
    out = provider.attach(df, "batt")
    ref = compute_battery_availability(
        df,
        component_name="batt",
        battery_power_column="batt__power_kw",
        battery_soc_column="batt__soc",
        battery_rated_power=100.0,
        battery_rated_energy=200.0,
        battery_min_soc=0.1,
        eta_discharge=0.95,
    )
    np.testing.assert_allclose(
        out["batt__availability"].to_numpy(),
        ref["batt_availability"].to_numpy(),
    )

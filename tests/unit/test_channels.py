"""Unit tests for the L1 channel builders, with hand-computed expected values."""

import numpy as np
import pandas as pd

from herc_analysis import channels


def test_kw_to_mw():
    out = channels.kw_to_mw(pd.Series([0.0, 1000.0, 18000.0]))
    np.testing.assert_allclose(out, [0.0, 1.0, 18.0])


def test_energy_mwh_single_step():
    # 18000 kW over 2 s = 18000 * 2 / 3600 / 1000 = 0.01 MWh
    out = channels.energy_mwh(pd.Series([18000.0]), dt_s=2.0)
    np.testing.assert_allclose(out, [0.01])


def test_energy_mwh_matches_two_step_form():
    # energy_mwh(power_kw, dt) == (power_kw / 1000) * dt / 3600  [old form]
    power_kw = pd.Series([5000.0, 10000.0, 2000.0])
    dt_s = 2.0
    old = (power_kw / 1000.0) * dt_s / 3600.0
    np.testing.assert_allclose(channels.energy_mwh(power_kw, dt_s), old)


def test_revenue_usd():
    price = pd.Series([10.0, 0.0, -5.0])
    energy = pd.Series([2.0, 3.0, 4.0])
    np.testing.assert_allclose(channels.revenue_usd(price, energy), [20.0, 0.0, -20.0])


def test_charge_discharge_split():
    energy = pd.Series([2.0, -3.0, 0.0, 5.0, -1.0])
    discharge, charge = channels.charge_discharge_split(energy)
    np.testing.assert_allclose(discharge, [2.0, 0.0, 0.0, 5.0, 0.0])
    np.testing.assert_allclose(charge, [0.0, -3.0, 0.0, 0.0, -1.0])
    # Each element lands in exactly one bucket; the two recombine to the input.
    np.testing.assert_allclose(discharge + charge, energy)


def test_soc_mileage():
    # |0.5-1.0| + |0.5-0.5| + |0.9-0.5| = 0.5 + 0.0 + 0.4 = 0.9
    soc = pd.Series([1.0, 0.5, 0.5, 0.9])
    assert channels.soc_mileage(soc) == 0.9


def test_soc_mileage_constant_is_zero():
    assert channels.soc_mileage(pd.Series([1.0, 1.0, 1.0])) == 0.0


def test_hourly_mean_broadcasts_group_mean():
    time_utc = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 00:30:00",
            "2024-01-01 01:00:00",
        ],
        utc=True,
    )
    values = pd.Series([10.0, 20.0, 7.0])
    out = channels.hourly_mean(values, pd.Series(time_utc))
    # First two share hour 00 -> mean 15; third is alone in hour 01 -> 7.
    np.testing.assert_allclose(out, [15.0, 15.0, 7.0])

"""Unit tests for the L1 series->scalar reducers, with hand-computed values."""

import math

import pandas as pd

from herc_analysis import reducers


def test_total_and_mean():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert reducers.total(s) == 10.0
    assert reducers.mean(s) == 2.5


def test_capacity_factor_known_value():
    # Fixture analogue: 18 MW plant, 22 MW interconnect, 1 h -> 18 MWh / 22.
    assert reducers.capacity_factor(18.0, 22.0, 1.0) == 18.0 / 22.0


def test_capacity_factor_zero_denominator_is_nan():
    assert math.isnan(reducers.capacity_factor(5.0, 0.0, 10.0))
    assert math.isnan(reducers.capacity_factor(5.0, 10.0, 0.0))


def test_value_factor_known_value():
    # revenue 120, avg_price 10, energy 10 -> base 100 -> 1.2
    assert reducers.value_factor(120.0, 10.0, 10.0) == 1.2


def test_value_factor_guards_to_nan():
    assert math.isnan(reducers.value_factor(0.0, 10.0, 10.0))  # revenue <= 0
    assert math.isnan(reducers.value_factor(120.0, 0.0, 10.0))  # base <= 0
    assert math.isnan(reducers.value_factor(120.0, 10.0, 0.0))  # base <= 0


def test_tb4_spread():
    # 8 hourly points; top-2 mean - bottom-2 mean with n_pts_4h=2.
    # top 2 = {80, 70} -> 75 ; bottom 2 = {10, 20} -> 15 ; spread = 60.
    price = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    assert reducers.tb4_spread(price, n_pts_4h=2) == 60.0


def test_tb4_spread_flat_is_zero():
    price = pd.Series([25.0] * 6)
    assert reducers.tb4_spread(price, n_pts_4h=2) == 0.0

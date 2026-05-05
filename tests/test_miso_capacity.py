"""Tests for herc_analysis.miso_capacity."""

import numpy as np
import pandas as pd
import pytest

from herc_analysis.miso_capacity import limit_hourly_availability_contributions


def _make_df(times, values_by_column):
    """Build a small test frame with a timezone-aware ``time_utc`` column."""
    data = {"time_utc": pd.to_datetime(times, utc=True)}
    data.update(values_by_column)
    return pd.DataFrame(data)


def test_limit_returns_input_when_below_limit():
    df = _make_df(
        [
            "2024-01-01 00:00",
            "2024-01-01 00:15",
            "2024-01-01 00:30",
            "2024-01-01 00:45",
        ],
        {"a": [1.0, 1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0, 2.0]},
    )
    result = limit_hourly_availability_contributions(df, ["a", "b"], limit=10.0)
    pd.testing.assert_frame_equal(result, df)


def test_limit_pro_rata_scales_columns_equally_within_hour():
    df = _make_df(
        ["2024-01-01 00:00", "2024-01-01 00:30"],
        {"a": [4.0, 4.0], "b": [4.0, 4.0]},
    )
    result = limit_hourly_availability_contributions(
        df, ["a", "b"], limit=4.0, priority_order=False
    )
    expected = _make_df(
        ["2024-01-01 00:00", "2024-01-01 00:30"],
        {"a": [2.0, 2.0], "b": [2.0, 2.0]},
    )
    pd.testing.assert_frame_equal(result, expected)


def test_limit_pro_rata_preserves_subhourly_shape():
    df = _make_df(
        [
            "2024-01-01 00:00",
            "2024-01-01 00:15",
            "2024-01-01 00:30",
            "2024-01-01 00:45",
        ],
        {"a": [0.0, 4.0, 8.0, 12.0], "b": [12.0, 8.0, 4.0, 0.0]},
    )
    # Hourly mean of total = mean([12, 12, 12, 12]) = 12 -> scale 0.5 to hit limit 6
    result = limit_hourly_availability_contributions(
        df, ["a", "b"], limit=6.0, priority_order=False
    )
    np.testing.assert_allclose(result["a"].to_numpy(), [0.0, 2.0, 4.0, 6.0])
    np.testing.assert_allclose(result["b"].to_numpy(), [6.0, 4.0, 2.0, 0.0])
    # The new hourly mean of the total is exactly the limit.
    assert (result["a"] + result["b"]).mean() == pytest.approx(6.0)


def test_limit_priority_order_reduces_last_column_first():
    df = _make_df(
        ["2024-01-01 00:00", "2024-01-01 00:30"],
        {"a": [4.0, 4.0], "b": [4.0, 4.0]},
    )
    # Hourly mean of total = 8, excess = 2.  Last column ("b") absorbs all
    # of the reduction; "a" is untouched.
    result = limit_hourly_availability_contributions(
        df, ["a", "b"], limit=6.0, priority_order=True
    )
    np.testing.assert_allclose(result["a"].to_numpy(), [4.0, 4.0])
    np.testing.assert_allclose(result["b"].to_numpy(), [2.0, 2.0])


def test_limit_priority_order_cascades_when_last_column_zeroed():
    df = _make_df(
        ["2024-01-01 00:00", "2024-01-01 00:30"],
        {"a": [4.0, 4.0], "b": [1.0, 1.0]},
    )
    # Hourly mean of total = 5, excess = 3.  "b" can supply only 1 (its
    # full hourly mean); the remaining 2 must come from "a".
    result = limit_hourly_availability_contributions(
        df, ["a", "b"], limit=2.0, priority_order=True
    )
    np.testing.assert_allclose(result["b"].to_numpy(), [0.0, 0.0])
    np.testing.assert_allclose(result["a"].to_numpy(), [2.0, 2.0])
    assert (result["a"] + result["b"]).mean() == pytest.approx(2.0)


def test_limit_priority_order_preserves_subhourly_shape():
    df = _make_df(
        [
            "2024-01-01 00:00",
            "2024-01-01 00:15",
            "2024-01-01 00:30",
            "2024-01-01 00:45",
        ],
        {"a": [2.0, 2.0, 2.0, 2.0], "b": [0.0, 4.0, 8.0, 12.0]},
    )
    # hourly mean of b = 6, total mean = 8, excess = 4 -> b scales by 1 - 4/6 = 1/3.
    result = limit_hourly_availability_contributions(
        df, ["a", "b"], limit=4.0, priority_order=True
    )
    np.testing.assert_allclose(result["a"].to_numpy(), [2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(
        result["b"].to_numpy(), np.array([0.0, 4.0, 8.0, 12.0]) / 3.0
    )
    assert (result["a"] + result["b"]).mean() == pytest.approx(4.0)


def test_limit_treats_each_hour_independently():
    df = _make_df(
        [
            "2024-01-01 00:00",
            "2024-01-01 00:30",
            "2024-01-01 01:00",
            "2024-01-01 01:30",
        ],
        {"a": [1.0, 1.0, 5.0, 5.0], "b": [1.0, 1.0, 5.0, 5.0]},
    )
    # Hour 0: mean total = 2 (no limiting). Hour 1: mean total = 10, excess 6
    # -> reduce "b" by 5 (cap), then "a" by 1.
    result = limit_hourly_availability_contributions(
        df, ["a", "b"], limit=4.0, priority_order=True
    )
    np.testing.assert_allclose(result["a"].to_numpy(), [1.0, 1.0, 4.0, 4.0])
    np.testing.assert_allclose(result["b"].to_numpy(), [1.0, 1.0, 0.0, 0.0])


def test_limit_does_not_collapse_same_hour_across_days():
    # Six 10-minute samples per hour, on two different days at the same hour
    # of day.  Each day's hour must be limited independently rather than
    # being lumped into a single 12-sample group.
    times = [
        # Day 1, hour 00 -- mean total = 12, needs limiting to 6.
        "2024-01-01 00:00",
        "2024-01-01 00:10",
        "2024-01-01 00:20",
        "2024-01-01 00:30",
        "2024-01-01 00:40",
        "2024-01-01 00:50",
        # Day 2, hour 00 -- mean total = 2, well under the limit.
        "2024-01-02 00:00",
        "2024-01-02 00:10",
        "2024-01-02 00:20",
        "2024-01-02 00:30",
        "2024-01-02 00:40",
        "2024-01-02 00:50",
    ]
    df = _make_df(
        times,
        {
            "a": [6.0] * 6 + [1.0] * 6,
            "b": [6.0] * 6 + [1.0] * 6,
        },
    )
    result = limit_hourly_availability_contributions(
        df, ["a", "b"], limit=6.0, priority_order=False
    )
    # Day 1 scaled by 0.5; Day 2 untouched.
    np.testing.assert_allclose(result["a"].to_numpy(), [3.0] * 6 + [1.0] * 6)
    np.testing.assert_allclose(result["b"].to_numpy(), [3.0] * 6 + [1.0] * 6)


def test_limit_does_not_mutate_input():
    df = _make_df(
        ["2024-01-01 00:00", "2024-01-01 00:30"],
        {"a": [4.0, 4.0], "b": [4.0, 4.0]},
    )
    original = df.copy()
    _ = limit_hourly_availability_contributions(df, ["a", "b"], limit=2.0)
    pd.testing.assert_frame_equal(df, original)


def test_limit_handles_zero_mean_column_in_priority_order():
    df = _make_df(
        ["2024-01-01 00:00", "2024-01-01 00:30"],
        {"a": [4.0, 4.0], "b": [0.0, 0.0]},
    )
    # "b" has zero hourly mean, so all excess must come from "a".
    result = limit_hourly_availability_contributions(
        df, ["a", "b"], limit=1.0, priority_order=True
    )
    np.testing.assert_allclose(result["a"].to_numpy(), [1.0, 1.0])
    np.testing.assert_allclose(result["b"].to_numpy(), [0.0, 0.0])


def test_limit_raises_when_time_utc_missing():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="time_utc"):
        limit_hourly_availability_contributions(df, ["a"], limit=1.0)


def test_limit_raises_when_availability_column_missing():
    df = _make_df(["2024-01-01 00:00"], {"a": [1.0]})
    with pytest.raises(ValueError, match="'b'"):
        limit_hourly_availability_contributions(df, ["a", "b"], limit=1.0)


def test_limit_raises_when_time_utc_not_tz_aware():
    df = pd.DataFrame({"time_utc": pd.to_datetime(["2024-01-01 00:00:00"]), "a": [1.0]})
    with pytest.raises(ValueError, match="timezone-aware"):
        limit_hourly_availability_contributions(df, ["a"], limit=1.0)


def test_limit_raises_when_time_utc_wrong_dtype():
    df = pd.DataFrame({"time_utc": ["2024-01-01 00:00:00"], "a": [1.0]})
    with pytest.raises(TypeError, match="datetime"):
        limit_hourly_availability_contributions(df, ["a"], limit=1.0)

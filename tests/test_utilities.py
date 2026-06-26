"""Tests for herc_analysis.utilities."""

import pandas as pd
import pytest

from herc_analysis.timeseries import add_local_time

# New York City coordinates -> America/New_York timezone.
NYC_LAT = 40.7128
NYC_LON = -74.0060


def test_add_local_time_appends_local_time_column():
    df = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(
                ["2024-01-01 12:00:00", "2024-06-01 12:00:00"], utc=True
            ),
            "value": [1.0, 2.0],
        }
    )
    result = add_local_time(df, latitude=NYC_LAT, longitude=NYC_LON)

    assert "time_local" in result.columns
    assert str(result["time_local"].dt.tz) == "America/New_York"


def test_add_local_time_converts_utc_to_local_correctly():
    df = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(
                ["2024-01-01 12:00:00", "2024-06-01 12:00:00"], utc=True
            )
        }
    )
    result = add_local_time(df, latitude=NYC_LAT, longitude=NYC_LON)

    expected = pd.to_datetime(
        ["2024-01-01 12:00:00", "2024-06-01 12:00:00"], utc=True
    ).tz_convert("America/New_York")
    pd.testing.assert_series_equal(
        result["time_local"].reset_index(drop=True),
        pd.Series(expected, name="time_local"),
    )


def test_add_local_time_does_not_mutate_input():
    df = pd.DataFrame({"time_utc": pd.to_datetime(["2024-01-01 00:00:00"], utc=True)})
    _ = add_local_time(df, latitude=NYC_LAT, longitude=NYC_LON)
    assert "time_local" not in df.columns


def test_add_local_time_raises_when_time_utc_missing():
    df = pd.DataFrame({"value": [1.0]})
    with pytest.raises(ValueError, match="time_utc"):
        add_local_time(df, latitude=NYC_LAT, longitude=NYC_LON)


def test_add_local_time_raises_when_time_utc_not_tz_aware():
    df = pd.DataFrame({"time_utc": pd.to_datetime(["2024-01-01 00:00:00"])})
    with pytest.raises(ValueError, match="timezone-aware"):
        add_local_time(df, latitude=NYC_LAT, longitude=NYC_LON)


def test_add_local_time_raises_when_time_utc_wrong_dtype():
    df = pd.DataFrame({"time_utc": ["2024-01-01 00:00:00"]})
    with pytest.raises(TypeError, match="datetime"):
        add_local_time(df, latitude=NYC_LAT, longitude=NYC_LON)

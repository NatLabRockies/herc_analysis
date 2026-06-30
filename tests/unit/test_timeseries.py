"""Unit tests for the new L1 timeseries primitives.

``interpolate_df`` / ``add_local_time`` keep their existing coverage in
``tests/test_utilities.py`` (now importing them from ``herc_analysis.timeseries``);
here we test the newly added primitives.
"""

import numpy as np
import pandas as pd

from herc_analysis import reducers, timeseries


def test_planning_year_sept_boundary():
    # Sept 1 05:00 UTC is the boundary (Sept 1 00:00 EST).
    times = pd.to_datetime(
        [
            "2022-08-31 23:00:00",  # before -> 2021/22 PY = 2122
            "2022-09-01 05:00:00",  # at/after -> 2022/23 PY = 2223
            "2023-01-15 12:00:00",  # still 2022/23 -> 2223
        ],
        utc=True,
    )
    out = timeseries.planning_year(pd.Series(times))
    assert list(out) == [2122, 2223, 2223]


def test_planning_year_matches_miso_shim():
    from herc_analysis.capacity._miso_engine import _time_to_planning_year

    times = pd.Series(
        pd.to_datetime(
            ["2021-09-01 05:00:00", "2024-03-10 18:00:00", "2020-12-31 23:30:00"],
            utc=True,
        )
    )
    pd.testing.assert_series_equal(
        timeseries.planning_year(times), _time_to_planning_year(times)
    )


def test_to_hourly_floors_and_means():
    df = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:30:00",
                    "2024-01-01 01:15:00",
                ],
                utc=True,
            ),
            "p": [10.0, 20.0, 7.0],
        }
    )
    out = timeseries.to_hourly(df, ["p"], how="mean")
    np.testing.assert_allclose(out["p"].to_numpy(), [15.0, 7.0])
    assert len(out) == 2


def test_period_labels_total_yearly_monthly():
    times = pd.Series(
        pd.to_datetime(
            ["2024-01-15 00:00:00", "2024-03-20 00:00:00", "2025-02-01 00:00:00"],
            utc=True,
        )
    )
    assert list(timeseries.period_labels(times, "total")) == ["total"] * 3
    assert list(timeseries.period_labels(times, "yearly")) == ["2024", "2024", "2025"]
    assert list(timeseries.period_labels(times, "monthly")) == [
        "2024-01",
        "2024-03",
        "2025-02",
    ]


def test_period_labels_rejects_unknown():
    times = pd.Series(pd.to_datetime(["2024-01-01"], utc=True))
    try:
        timeseries.period_labels(times, "weekly")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown resolution")


def test_aggregate_metric_by_month():
    times = pd.Series(
        pd.to_datetime(
            ["2024-01-01", "2024-01-15", "2024-02-01"],
            utc=True,
        )
    )
    values = pd.Series([1.0, 2.0, 5.0])
    out = timeseries.aggregate_metric(values, times, "monthly", reducers.total)
    assert out["2024-01"] == 3.0
    assert out["2024-02"] == 5.0

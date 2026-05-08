"""Tests for herc_analysis.miso_capacity."""

import numpy as np
import pandas as pd

from herc_analysis.miso_capacity import (
    RA_HOURS_CSV_PATH,
    MisoCapacity,
)


def _make_hourly_df(start, n_hours, values_by_column):
    """Build a small hourly test frame with a tz-aware ``time_utc`` column."""
    data = {"time_utc": pd.date_range(start=start, periods=n_hours, freq="h", tz="UTC")}
    data.update(values_by_column)
    return pd.DataFrame(data)


def _make_capacity(
    df,
    interconnect_limit,
    priority_order=None,
    remove_low_hour_seasons=False,
    class_list=None,
):
    """Construct a ``MisoCapacity`` with sensible defaults for testing.

    ``remove_low_hour_seasons`` defaults to ``False`` here so test
    windows of just a few hours don't get filtered to empty before the
    metrics are computed.
    """
    component_list = [c for c in df.columns if c != "time_utc"]
    if class_list is None:
        class_list = ["wind"] * len(component_list)
    return MisoCapacity(
        component_list=component_list,
        class_list=class_list,
        df=df,
        zone=1,
        interconnect_limit=interconnect_limit,
        priority_order=priority_order,
        remove_low_hour_seasons=remove_low_hour_seasons,
    )


def test_limit_pro_rata_returns_input_when_below_limit():
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [1.0, 1.0], "b": [2.0, 2.0]},
    )
    mc = _make_capacity(df, interconnect_limit=10.0)
    np.testing.assert_allclose(mc.df_h_limit["a"].to_numpy(), [1.0, 1.0])
    np.testing.assert_allclose(mc.df_h_limit["b"].to_numpy(), [2.0, 2.0])


def test_limit_pro_rata_scales_columns_equally():
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [4.0, 4.0], "b": [4.0, 4.0]},
    )
    mc = _make_capacity(df, interconnect_limit=4.0)
    np.testing.assert_allclose(mc.df_h_limit["a"].to_numpy(), [2.0, 2.0])
    np.testing.assert_allclose(mc.df_h_limit["b"].to_numpy(), [2.0, 2.0])


def test_limit_pro_rata_treats_each_hour_independently():
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [1.0, 5.0], "b": [1.0, 5.0]},
    )
    # Hour 0: row_sum = 2 (under limit, untouched).
    # Hour 1: row_sum = 10 -> scale 4/10 -> each column becomes 2.
    mc = _make_capacity(df, interconnect_limit=4.0)
    np.testing.assert_allclose(mc.df_h_limit["a"].to_numpy(), [1.0, 2.0])
    np.testing.assert_allclose(mc.df_h_limit["b"].to_numpy(), [1.0, 2.0])


def test_limit_priority_order_list_reduces_last_first():
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [4.0, 4.0], "b": [4.0, 4.0]},
    )
    # row_sum = 8, excess = 2.  Last priority ("b") absorbs all of it.
    mc = _make_capacity(df, interconnect_limit=6.0, priority_order=["a", "b"])
    np.testing.assert_allclose(mc.df_h_limit["a"].to_numpy(), [4.0, 4.0])
    np.testing.assert_allclose(mc.df_h_limit["b"].to_numpy(), [2.0, 2.0])


def test_limit_priority_order_list_cascades_when_last_zeroed():
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [4.0, 4.0], "b": [1.0, 1.0]},
    )
    # row_sum = 5, limit = 2 -> excess = 3.  "b" provides 1, "a" provides 2.
    mc = _make_capacity(df, interconnect_limit=2.0, priority_order=["a", "b"])
    np.testing.assert_allclose(mc.df_h_limit["b"].to_numpy(), [0.0, 0.0])
    np.testing.assert_allclose(mc.df_h_limit["a"].to_numpy(), [2.0, 2.0])


def test_limit_priority_order_handles_zero_value_column():
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [4.0, 4.0], "b": [0.0, 0.0]},
    )
    # "b" has zero value, so all excess must come from "a".
    mc = _make_capacity(df, interconnect_limit=1.0, priority_order=["a", "b"])
    np.testing.assert_allclose(mc.df_h_limit["a"].to_numpy(), [1.0, 1.0])
    np.testing.assert_allclose(mc.df_h_limit["b"].to_numpy(), [0.0, 0.0])


def test_limit_priority_order_dict_varies_by_season():
    # One row in winter (Jan), one row in summer (Jul); each is 4 + 4.
    df = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(
                ["2024-01-01 00:00", "2024-07-01 00:00"], utc=True
            ),
            "a": [4.0, 4.0],
            "b": [4.0, 4.0],
        }
    )
    # Winter: cut "b" first; Summer: cut "a" first.
    priority_order = {
        "winter": ["a", "b"],
        "summer": ["b", "a"],
        "fall": ["a", "b"],
        "spring": ["a", "b"],
    }
    mc = _make_capacity(df, interconnect_limit=6.0, priority_order=priority_order)

    df_by_season = mc.df_h_limit.set_index("season")[["a", "b"]]
    # Each row has excess 2.
    np.testing.assert_allclose(df_by_season.loc["winter"].to_numpy(), [4.0, 2.0])
    np.testing.assert_allclose(df_by_season.loc["summer"].to_numpy(), [2.0, 4.0])


def test_limit_does_not_mutate_input_df():
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [4.0, 4.0], "b": [4.0, 4.0]},
    )
    original = df.copy()
    _ = _make_capacity(df, interconnect_limit=2.0)
    pd.testing.assert_frame_equal(df, original)


def test_limit_preserves_non_component_columns_in_df_h():
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [4.0, 4.0], "b": [4.0, 4.0]},
    )
    mc = _make_capacity(df, interconnect_limit=4.0)
    # The merged frame carries the season / RA flag columns; the limited
    # frame should keep them intact.
    pd.testing.assert_series_equal(
        mc.df_h_limit["season"], mc.df_h["season"], check_names=False
    )
    assert "ra_central_north" in mc.df_h_limit.columns
    assert "year" in mc.df_h_limit.columns


# ---------------------------------------------------------------------------
# availability metrics tests (computed in __init__)
# ---------------------------------------------------------------------------


def _availability_capacity():
    """Build a MisoCapacity over a small (zone 1) window with mixed flags.

    The window is 2022-09-01 05:00 UTC through 2022-09-03 04:00 UTC (48h),
    which is fully covered by the bundled RA-hour reference table and
    contains both RA-hour and AAOC-hour entries.  Two components are
    used so the per-component dict shape is exercised.
    """
    df = _make_hourly_df(
        "2022-09-01 05:00",
        48,
        {
            "a": np.linspace(1.0, 4.0, 48),
            "b": np.linspace(2.0, 5.0, 48),
        },
    )
    return _make_capacity(df, interconnect_limit=100.0)


def test_metrics_attributes_populated():
    mc = _availability_capacity()
    for name in (
        "aaoc_per_year",
        "hours_per_year_season",
        "tier_1_availability",
        "tier_2_availability",
        "all_hours_availability",
        "isac",
        "remove_low_hour_seasons",
    ):
        assert hasattr(mc, name), f"MisoCapacity is missing attribute {name!r}"


def test_metrics_hours_per_year_season_matches_df_h_limit():
    mc = _availability_capacity()
    df = mc.df_h_limit
    expected_per_year_season = {
        (int(y), str(s)): int(c)
        for (y, s), c in df.groupby(["year", "season"]).size().items()
    }
    assert mc.hours_per_year_season == expected_per_year_season


def test_metrics_aaoc_per_year_matches_direct_mean():
    mc = _availability_capacity()
    df = mc.df_h_limit
    aaoc_rows = df[df["aaoc_central_north"]]
    for (year, component), value in mc.aaoc_per_year.items():
        expected = float(aaoc_rows.loc[aaoc_rows["year"] == year, component].mean())
        np.testing.assert_allclose(value, expected)


def test_metrics_tier_1_and_all_hours_match_direct_means():
    mc = _availability_capacity()
    df = mc.df_h_limit
    ra_col = "ra_central_north"

    for season in df["season"].unique():
        df_season = df[df["season"] == season]
        for component in mc.component_list:
            expected_tier_1 = float(df_season.loc[~df_season[ra_col], component].mean())
            expected_all = float(df_season[component].mean())
            np.testing.assert_allclose(
                mc.tier_1_availability[(season, component)], expected_tier_1
            )
            np.testing.assert_allclose(
                mc.all_hours_availability[(season, component)], expected_all
            )


def test_metrics_isac_formula():
    mc = _availability_capacity()
    for key, isac_value in mc.isac.items():
        expected = 0.2 * mc.tier_1_availability[key] + 0.8 * mc.tier_2_availability[key]
        np.testing.assert_allclose(isac_value, expected)


def test_metrics_constant_availability_collapses_to_constant():
    # With every hourly availability equal to the same constant, every
    # tier mean (including the AAOC-padded tier_2) must equal the
    # constant for every (season, component) pair.
    constant = 3.5
    df = _make_hourly_df(
        "2022-09-01 05:00",
        48,
        {"a": [constant] * 48, "b": [constant] * 48},
    )
    mc = _make_capacity(df, interconnect_limit=100.0)

    for component in mc.component_list:
        for season in mc.df_h_limit["season"].unique():
            key = (season, component)
            np.testing.assert_allclose(mc.tier_1_availability[key], constant)
            np.testing.assert_allclose(mc.tier_2_availability[key], constant)
            np.testing.assert_allclose(mc.all_hours_availability[key], constant)
            np.testing.assert_allclose(mc.isac[key], constant)


def test_metrics_tier_2_pads_to_65_with_per_year_aaoc():
    # Build a 48-hour window over the start of the RA-hour reference
    # table (entirely in fall 2022).  Hours that are AAOC get
    # ``a = AAOC_VAL``, hours that are RA but not AAOC get ``a = RA_VAL``,
    # and other hours get an arbitrary value irrelevant to tier_2.  This
    # lets us derive the expected tier_2 mean analytically:
    #
    #     aaoc_per_year[(2022, 'a')] = AAOC_VAL
    #     tier_2_sum = (n_ra - n_aaoc) * RA_VAL + n_aaoc * AAOC_VAL
    #                  + (65 - n_ra) * AAOC_VAL
    #     tier_2_mean = tier_2_sum / 65
    AAOC_VAL = 2.0
    RA_VAL = 0.0
    OTHER_VAL = 9.0

    df_ra = pd.read_csv(RA_HOURS_CSV_PATH)
    df_ra["time_utc"] = pd.to_datetime(df_ra["time_utc"], utc=True)
    df_ra = df_ra.iloc[:48].copy()
    ra_flags = df_ra["ra_central_north"].astype(bool).to_numpy()
    aaoc_flags = df_ra["aaoc_central_north"].astype(bool).to_numpy()

    a_values = np.where(aaoc_flags, AAOC_VAL, np.where(ra_flags, RA_VAL, OTHER_VAL))
    df = pd.DataFrame(
        {
            "time_utc": df_ra["time_utc"].to_numpy(),
            "a": a_values,
            "b": np.full(48, 7.0),
        }
    )
    mc = _make_capacity(df, interconnect_limit=100.0)

    fall = "fall"
    n_ra = int(ra_flags.sum())
    n_aaoc = int(aaoc_flags.sum())
    assert 0 < n_ra < 65, "test window must have a partial RA-hour set"
    assert n_aaoc > 0, "test window must have at least one AAOC hour"

    np.testing.assert_allclose(mc.aaoc_per_year[(2022, "a")], AAOC_VAL)
    expected_tier_2_a = (
        (n_ra - n_aaoc) * RA_VAL + n_aaoc * AAOC_VAL + (65 - n_ra) * AAOC_VAL
    ) / 65
    np.testing.assert_allclose(mc.tier_2_availability[(fall, "a")], expected_tier_2_a)
    # Constant-7 control component should still come out at 7.0.
    np.testing.assert_allclose(mc.tier_2_availability[(fall, "b")], 7.0)


def test_remove_low_hour_seasons_default_is_true():
    df = _make_hourly_df("2022-09-01 05:00", 2, {"a": [1.0, 1.0], "b": [2.0, 2.0]})
    component_list = [c for c in df.columns if c != "time_utc"]
    mc = MisoCapacity(
        component_list=component_list,
        class_list=["wind"] * len(component_list),
        df=df,
        zone=1,
        interconnect_limit=10.0,
    )
    assert mc.remove_low_hour_seasons is True


def test_remove_low_hour_seasons_false_keeps_short_window():
    # A 48-hour window covers only ~2 days of fall; with the flag off
    # the metrics should still be computed, with df_h_limit retained.
    df = _make_hourly_df("2022-09-01 05:00", 48, {"a": np.linspace(1.0, 4.0, 48)})
    mc = _make_capacity(df, interconnect_limit=100.0, remove_low_hour_seasons=False)
    assert len(mc.df_h_limit) == 48
    # hours_per_year_season reflects the original (year, season).
    assert mc.hours_per_year_season == {(2022, "fall"): 48}
    # And the per-(season, component) availability dicts are populated.
    assert ("fall", "a") in mc.tier_1_availability
    assert ("fall", "a") in mc.isac


def test_remove_low_hour_seasons_true_drops_below_threshold():
    # Same 48-hour window, but with the flag on every (year, season)
    # pair has < 85 days of data and should be dropped.
    df = _make_hourly_df("2022-09-01 05:00", 48, {"a": np.linspace(1.0, 4.0, 48)})
    mc = _make_capacity(df, interconnect_limit=100.0, remove_low_hour_seasons=True)
    # df_h_limit should have been emptied, but the original counts
    # remain visible in hours_per_year_season for diagnostics.
    assert mc.df_h_limit.empty
    assert mc.hours_per_year_season == {(2022, "fall"): 48}
    # No tier values are produced because no (season, component) pair
    # survives the filter.
    assert mc.tier_1_availability == {}
    assert mc.tier_2_availability == {}
    assert mc.all_hours_availability == {}
    assert mc.isac == {}


def test_remove_low_hour_seasons_threshold_boundary():
    # 85 days exactly is kept; one hour less is dropped.
    threshold_hours = 85 * 24
    df_kept = _make_hourly_df(
        "2022-09-01 05:00",
        threshold_hours,
        {"a": np.zeros(threshold_hours)},
    )
    mc_kept = _make_capacity(
        df_kept, interconnect_limit=100.0, remove_low_hour_seasons=True
    )
    assert len(mc_kept.df_h_limit) == threshold_hours

    df_dropped = _make_hourly_df(
        "2022-09-01 05:00",
        threshold_hours - 1,
        {"a": np.zeros(threshold_hours - 1)},
    )
    mc_dropped = _make_capacity(
        df_dropped, interconnect_limit=100.0, remove_low_hour_seasons=True
    )
    assert mc_dropped.df_h_limit.empty

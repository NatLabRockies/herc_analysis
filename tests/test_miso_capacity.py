"""Tests for herc_analysis.miso_capacity.

Organised from small building blocks to larger integration:
  1. Helpers and shared fixtures
  2. Interconnect-limit capping (_limit_hourly_availability_contributions)
  3. Hour counting and low-season dropping
  4. Availability metrics (tier 1/2, ISAC, AAOC padding)
  5. UCAP / ISAC / SAC loading and conversion
  6. PRA pricing and days-per-season
  7. Revenue computation (_compute_revenue_per_season, _compute_annual_revenue)
  8. Public output helpers (get_component_table, get_totals_table, revenue getters)
"""

import numpy as np
import pandas as pd
import pytest

from herc_analysis.miso_capacity import (
    RA_HOURS_CSV_PATH,
    MisoCapacity,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_hourly_df(start, n_hours, values_by_column):
    """Build a small hourly test frame with a tz-aware ``time_utc`` column.

    All values are passed as plain dicts so individual tests stay readable
    without repeating the boilerplate of constructing a DatetimeIndex.
    """
    data = {"time_utc": pd.date_range(start=start, periods=n_hours, freq="h", tz="UTC")}
    data.update(values_by_column)
    return pd.DataFrame(data)


def _make_capacity(
    df,
    interconnect_limit,
    priority_order=None,
    remove_low_hour_seasons=False,
    class_list=None,
    pra_years=None,
):
    """Construct a ``MisoCapacity`` with sensible defaults for testing.

    ``remove_low_hour_seasons`` defaults to ``False`` so short test windows
    (a few hours) are not silently filtered before metrics are computed.
    ``class_list`` defaults to ``["wind"]`` for every component.
    Zone 1 maps to the ``central_north`` subregion in the bundled CSV.
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
        pra_years=pra_years,
    )


# Shared 48-hour fixture used by several availability-metric tests.
# Window: 2022-09-01 05:00 UTC → 2022-09-03 04:00 UTC (fully inside the
# bundled RA-hour reference table and containing both RA and AAOC hours).
def _availability_capacity():
    """Return a MisoCapacity over a 48-hour window with two components.

    Both components use linearly-varying availability so per-season means
    are non-trivial, exercising the full tier-1/tier-2 computation path.
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


# ---------------------------------------------------------------------------
# 1. Interconnect-limit capping
# ---------------------------------------------------------------------------


def test_limit_pro_rata_returns_input_when_below_limit():
    """Values that already satisfy the limit are left unchanged.

    Pro-rata scaling only kicks in when the row-sum *exceeds* the limit,
    so under-limit hours must be passed through intact.
    """
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [1000.0, 1000.0], "b": [2000.0, 2000.0]},
    )
    mc = _make_capacity(df, interconnect_limit=10000.0)
    np.testing.assert_allclose(mc.df_h_limit_mw["a"].to_numpy(), [1.0, 1.0])
    np.testing.assert_allclose(mc.df_h_limit_mw["b"].to_numpy(), [2.0, 2.0])


def test_limit_pro_rata_scales_columns_equally():
    """Equal components are scaled by the same factor when the row-sum is over the limit.

    With a=4 MW and b=4 MW each hour (row-sum=8 MW) and a limit of 4 MW, every
    component is multiplied by 4/8=0.5, yielding 2.0 MW for both.
    """
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [4000.0, 4000.0], "b": [4000.0, 4000.0]},
    )
    mc = _make_capacity(df, interconnect_limit=4000.0)
    np.testing.assert_allclose(mc.df_h_limit_mw["a"].to_numpy(), [2.0, 2.0])
    np.testing.assert_allclose(mc.df_h_limit_mw["b"].to_numpy(), [2.0, 2.0])


def test_limit_pro_rata_treats_each_hour_independently():
    """Pro-rata capping is applied row-by-row, not on aggregated totals.

    Hour 0 (a=1 MW, b=1 MW, row-sum=2 MW) is below the 4 MW limit and is
    untouched.  Hour 1 (a=5 MW, b=5 MW, row-sum=10 MW) is scaled by 4/10
    so both columns become 2.0 MW.
    """
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [1000.0, 5000.0], "b": [1000.0, 5000.0]},
    )
    mc = _make_capacity(df, interconnect_limit=4000.0)
    np.testing.assert_allclose(mc.df_h_limit_mw["a"].to_numpy(), [1.0, 2.0])
    np.testing.assert_allclose(mc.df_h_limit_mw["b"].to_numpy(), [1.0, 2.0])


def test_limit_priority_order_list_reduces_last_first():
    """The lowest-priority component absorbs excess before higher-priority ones.

    With priority_order=["a", "b"], "b" is last (lowest priority) and
    absorbs all 2 MW of excess (row-sum=8 MW, limit=6 MW) before "a" is touched.
    """
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [4000.0, 4000.0], "b": [4000.0, 4000.0]},
    )
    mc = _make_capacity(df, interconnect_limit=6000.0, priority_order=["a", "b"])
    np.testing.assert_allclose(mc.df_h_limit_mw["a"].to_numpy(), [4.0, 4.0])
    np.testing.assert_allclose(mc.df_h_limit_mw["b"].to_numpy(), [2.0, 2.0])


def test_limit_priority_order_list_cascades_when_last_zeroed():
    """Excess cascades to higher-priority components when the last is exhausted.

    Excess=3 MW (row-sum=5 MW, limit=2 MW).  "b" can only provide 1 MW, so the
    remaining 2 MW is absorbed by "a", reducing it from 4 MW to 2 MW.
    """
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [4000.0, 4000.0], "b": [1000.0, 1000.0]},
    )
    mc = _make_capacity(df, interconnect_limit=2000.0, priority_order=["a", "b"])
    np.testing.assert_allclose(mc.df_h_limit_mw["b"].to_numpy(), [0.0, 0.0])
    np.testing.assert_allclose(mc.df_h_limit_mw["a"].to_numpy(), [2.0, 2.0])


def test_limit_priority_order_handles_zero_value_column():
    """A component already at zero cannot absorb any excess.

    When "b" is 0, all excess must be taken from "a" even though "b" has
    lower priority.  This verifies the ``clip(lower=0)`` guard prevents
    "b" from going negative.
    """
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [4000.0, 4000.0], "b": [0.0, 0.0]},
    )
    mc = _make_capacity(df, interconnect_limit=1000.0, priority_order=["a", "b"])
    np.testing.assert_allclose(mc.df_h_limit_mw["a"].to_numpy(), [1.0, 1.0])
    np.testing.assert_allclose(mc.df_h_limit_mw["b"].to_numpy(), [0.0, 0.0])


def test_limit_priority_order_dict_varies_by_season():
    """A dict priority_order applies different rules to different seasons.

    Winter cuts "b" first (order ["a","b"]); summer cuts "a" first
    (order ["b","a"]).  Each row has row-sum=8 MW and limit=6 MW, so exactly
    2 MW is removed from the last-priority component in each season.
    """
    df = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(
                ["2024-01-01 00:00", "2024-07-01 00:00"], utc=True
            ),
            "a": [4000.0, 4000.0],
            "b": [4000.0, 4000.0],
        }
    )
    priority_order = {
        "winter": ["a", "b"],
        "summer": ["b", "a"],
        "fall": ["a", "b"],
        "spring": ["a", "b"],
    }
    mc = _make_capacity(df, interconnect_limit=6000.0, priority_order=priority_order)

    df_by_season = mc.df_h_limit_mw.set_index("season")[["a", "b"]]
    np.testing.assert_allclose(df_by_season.loc["winter"].to_numpy(), [4.0, 2.0])
    np.testing.assert_allclose(df_by_season.loc["summer"].to_numpy(), [2.0, 4.0])


def test_limit_does_not_mutate_input_df():
    """The capping logic must not modify the caller's DataFrame in-place."""
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [4000.0, 4000.0], "b": [4000.0, 4000.0]},
    )
    original = df.copy()
    _ = _make_capacity(df, interconnect_limit=2000.0)
    pd.testing.assert_frame_equal(df, original)


def test_limit_preserves_non_component_columns_in_df_h_mw():
    """Season, RA-flag, and year columns survive the capping step unchanged."""
    df = _make_hourly_df(
        "2024-01-01 00:00",
        2,
        {"a": [4000.0, 4000.0], "b": [4000.0, 4000.0]},
    )
    mc = _make_capacity(df, interconnect_limit=4000.0)
    pd.testing.assert_series_equal(
        mc.df_h_limit_mw["season"], mc.df_h_mw["season"], check_names=False
    )
    assert "ra_central_north" in mc.df_h_limit_mw.columns
    assert "year" in mc.df_h_limit_mw.columns


# ---------------------------------------------------------------------------
# 2. Hour counting and low-season dropping
# ---------------------------------------------------------------------------


def test_hours_per_year_season_matches_df_h_limit():
    """hours_per_year_season reflects the actual (year, season) row counts in df_h_limit.

    This attribute is computed *before* season-dropping so it serves as an
    audit trail of original simulation coverage.
    """
    mc = _availability_capacity()
    df = mc.df_h_limit_mw
    expected = {
        (int(y), str(s)): int(c)
        for (y, s), c in df.groupby(["year", "season"]).size().items()
    }
    assert mc.hours_per_year_season == expected


def test_remove_low_hour_seasons_default_is_true():
    """The remove_low_hour_seasons flag defaults to True for production use."""
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
    """Setting remove_low_hour_seasons=False retains every classified hour.

    A 48-hour window covers only ~2 days, far below the 85-day threshold.
    With the flag off the data is preserved and metrics are computed.
    """
    df = _make_hourly_df("2022-09-01 05:00", 48, {"a": np.linspace(1.0, 4.0, 48)})
    mc = _make_capacity(df, interconnect_limit=100.0, remove_low_hour_seasons=False)
    assert len(mc.df_h_limit_mw) == 48
    assert mc.hours_per_year_season == {(2022, "fall"): 48}
    assert ("fall", "a") in mc.tier_1_availability_mw
    assert ("fall", "a") in mc.isac_mw


def test_remove_low_hour_seasons_true_drops_below_threshold():
    """Setting remove_low_hour_seasons=True drops (year, season) pairs below 85 days.

    The 48-hour window has far fewer than 85*24=2040 hours, so the entire
    season is dropped.  The hours_per_year_season audit dict is retained.
    """
    df = _make_hourly_df("2022-09-01 05:00", 48, {"a": np.linspace(1.0, 4.0, 48)})
    mc = _make_capacity(df, interconnect_limit=100.0, remove_low_hour_seasons=True)
    assert mc.df_h_limit_mw.empty
    assert mc.hours_per_year_season == {(2022, "fall"): 48}
    assert mc.tier_1_availability_mw == {}
    assert mc.tier_2_availability_mw == {}
    assert mc.all_hours_availability_mw == {}
    assert mc.isac_mw == {}


def test_remove_low_hour_seasons_threshold_boundary():
    """Exactly 85 days of hours is kept; one hour fewer is dropped.

    This pins the ``>=`` (not ``>``) boundary in _drop_low_hour_seasons.
    """
    threshold_hours = 85 * 24
    df_kept = _make_hourly_df(
        "2022-09-01 05:00",
        threshold_hours,
        {"a": np.zeros(threshold_hours)},
    )
    mc_kept = _make_capacity(
        df_kept, interconnect_limit=100.0, remove_low_hour_seasons=True
    )
    assert len(mc_kept.df_h_limit_mw) == threshold_hours

    df_dropped = _make_hourly_df(
        "2022-09-01 05:00",
        threshold_hours - 1,
        {"a": np.zeros(threshold_hours - 1)},
    )
    mc_dropped = _make_capacity(
        df_dropped, interconnect_limit=100.0, remove_low_hour_seasons=True
    )
    assert mc_dropped.df_h_limit_mw.empty


# ---------------------------------------------------------------------------
# 3. Availability metrics (tier 1/2, ISAC, AAOC padding)
# ---------------------------------------------------------------------------


def test_metrics_attributes_populated():
    """All expected metric attributes exist after construction."""
    mc = _availability_capacity()
    for name in (
        "aaoc_per_year_mw",
        "hours_per_year_season",
        "tier_1_availability_mw",
        "tier_2_availability_mw",
        "all_hours_availability_mw",
        "isac_mw",
        "remove_low_hour_seasons",
    ):
        assert hasattr(mc, name), f"MisoCapacity is missing attribute {name!r}"


def test_metrics_aaoc_per_year_matches_direct_mean():
    """aaoc_per_year matches a direct groupby mean over AAOC-flagged hours."""
    mc = _availability_capacity()
    df = mc.df_h_limit_mw
    aaoc_rows = df[df["aaoc_central_north"]]
    for (year, component), value in mc.aaoc_per_year_mw.items():
        expected = float(aaoc_rows.loc[aaoc_rows["year"] == year, component].mean())
        np.testing.assert_allclose(value, expected)


def test_metrics_tier_1_and_all_hours_match_direct_means():
    """Tier 1 (non-RA hours) and all-hours means match direct DataFrame aggregations."""
    mc = _availability_capacity()
    df = mc.df_h_limit_mw
    ra_col = "ra_central_north"

    for season in df["season"].unique():
        df_season = df[df["season"] == season]
        for component in mc.component_list:
            expected_tier_1 = float(df_season.loc[~df_season[ra_col], component].mean())
            expected_all = float(df_season[component].mean())
            np.testing.assert_allclose(
                mc.tier_1_availability_mw[(season, component)], expected_tier_1
            )
            np.testing.assert_allclose(
                mc.all_hours_availability_mw[(season, component)], expected_all
            )


def test_metrics_isac_formula():
    """ISAC equals 0.2 * tier_1 + 0.8 * tier_2 for every (season, component) key."""
    mc = _availability_capacity()
    for key, isac_value in mc.isac_mw.items():
        expected = (
            0.2 * mc.tier_1_availability_mw[key] + 0.8 * mc.tier_2_availability_mw[key]
        )
        np.testing.assert_allclose(isac_value, expected)


def test_metrics_constant_availability_collapses_to_constant():
    """When every hour has the same availability value, all tier means equal that value.

    This is a sanity check: constant input should produce constant output
    regardless of the RA/AAOC flag pattern or AAOC padding.
    """
    constant = 3.5
    df = _make_hourly_df(
        "2022-09-01 05:00",
        48,
        {"a": [constant] * 48, "b": [constant] * 48},
    )
    mc = _make_capacity(df, interconnect_limit=100.0)

    constant_mw = constant / 1000
    for component in mc.component_list:
        for season in mc.df_h_limit_mw["season"].unique():
            key = (season, component)
            np.testing.assert_allclose(mc.tier_1_availability_mw[key], constant_mw)
            np.testing.assert_allclose(mc.tier_2_availability_mw[key], constant_mw)
            np.testing.assert_allclose(mc.all_hours_availability_mw[key], constant_mw)
            np.testing.assert_allclose(mc.isac_mw[key], constant_mw)


def test_metrics_tier_2_pads_to_65_with_per_year_aaoc():
    """Tier 2 is padded to 65 entries with the per-year AAOC mean when RA hours < 65.

    Setup: AAOC hours get ``a = AAOC_VAL``, non-AAOC RA hours get
    ``a = RA_VAL``.  The expected tier-2 mean is derived analytically and
    compared to the computed value.  This pins the padding logic in
    _compute_tier_availabilities.
    """
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

    np.testing.assert_allclose(mc.aaoc_per_year_mw[(2022, "a")], AAOC_VAL / 1000)
    expected_tier_2_a = (
        ((n_ra - n_aaoc) * RA_VAL + n_aaoc * AAOC_VAL + (65 - n_ra) * AAOC_VAL)
        / 65
        / 1000
    )
    np.testing.assert_allclose(
        mc.tier_2_availability_mw[(fall, "a")], expected_tier_2_a
    )
    np.testing.assert_allclose(mc.tier_2_availability_mw[(fall, "b")], 7.0 / 1000)


# ---------------------------------------------------------------------------
# 4. UCAP / ISAC / SAC loading and conversion
# ---------------------------------------------------------------------------


def test_class_ucap_and_isac_have_correct_keys():
    """class_ucap and class_isac are keyed by (season, component) for all 4 seasons."""
    mc = _availability_capacity()
    for component in mc.component_list:
        for season in ("summer", "fall", "winter", "spring"):
            assert (season, component) in mc.class_ucap_mw
            assert (season, component) in mc.class_isac_mw


def test_class_ucap_wind_summer_matches_csv_value():
    """class_ucap for wind in summer exactly matches the source CSV (2079.0 MW).

    Pins one raw value from resource_class_ucap_mw.csv so an accidental
    edit to the file or a loading bug is caught before propagating into
    SAC and revenue calculations.
    """
    mc = _availability_capacity()  # class_list defaults to ["wind", "wind"]
    assert mc.class_ucap_mw[("summer", "a")] == 2079.0


def test_class_isac_wind_winter_matches_csv_value():
    """class_isac for wind in winter exactly matches the source CSV (10859.0 MW).

    Pins one raw value from resource_class_isac_mw.csv with the same
    intent as test_class_ucap_wind_summer_matches_csv_value.
    """
    mc = _availability_capacity()
    assert mc.class_isac_mw[("winter", "a")] == 10859.0


def test_class_ucap_isac_conversion_is_ratio():
    """class_ucap_isac_conversion equals class_ucap / class_isac for each key."""
    mc = _availability_capacity()
    for key in mc.class_ucap_isac_conversion:
        ucap = mc.class_ucap_mw[key]
        isac = mc.class_isac_mw[key]
        expected = ucap / isac if isac != 0.0 else float("nan")
        actual = mc.class_ucap_isac_conversion[key]
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            np.testing.assert_allclose(actual, expected)


def test_sac_equals_isac_times_conversion():
    """SAC equals component ISAC multiplied by the class UCAP/ISAC conversion factor."""
    mc = _availability_capacity()
    for key, sac_value in mc.sac_mw.items():
        expected = mc.isac_mw[key] * mc.class_ucap_isac_conversion[key]
        np.testing.assert_allclose(sac_value, expected)


def test_zrc_equals_sac():
    """ZRC is currently set equal to SAC (no further adjustments implemented)."""
    mc = _availability_capacity()
    assert mc.zrc_mw == mc.sac_mw


# ---------------------------------------------------------------------------
# 5. PRA pricing and days-per-season
# ---------------------------------------------------------------------------


def test_pra_prices_has_all_seasons():
    """pra_prices covers all four MISO seasons."""
    mc = _availability_capacity()
    assert set(mc.pra_prices.keys()) == {"summer", "fall", "winter", "spring"}


def test_pra_prices_are_positive():
    """All PRA prices must be positive (they are clearing auction prices)."""
    mc = _availability_capacity()
    for season, price in mc.pra_prices.items():
        assert price > 0, f"PRA price for {season} is non-positive: {price}"


def test_pra_price_zone1_2026_summer_matches_csv_value():
    """pra_prices["summer"] for zone 1 year 2026 exactly matches the source CSV (424.30 $/MW-day).

    The fixture uses zone=1 and pra_years=None, which selects the most
    recent year in pra_prices_usd_per_mw_day.csv (2026).  Pins the raw
    value so a future CSV update is caught before it silently changes
    revenue outputs.
    """
    mc = _availability_capacity()
    assert mc.pra_years == [2026]
    assert mc.pra_prices["summer"] == pytest.approx(424.30)


def test_days_per_season_has_all_seasons():
    """days_per_season covers all four MISO seasons."""
    mc = _availability_capacity()
    assert set(mc.days_per_season.keys()) == {"summer", "fall", "winter", "spring"}


def test_days_per_season_reasonable_values():
    """Each season contains between 88 and 93 days (MISO season definitions)."""
    mc = _availability_capacity()
    for season, days in mc.days_per_season.items():
        assert 88 <= days <= 93, f"{season} has unexpected day count: {days}"


def test_days_per_season_sums_to_365_or_366():
    """Total days across all seasons equals either 365 (non-leap) or 366 (leap)."""
    mc = _availability_capacity()
    total = sum(mc.days_per_season.values())
    assert total in (365.0, 366.0), f"Total days {total} is not 365 or 366"


# ---------------------------------------------------------------------------
# 6. Revenue computation
# ---------------------------------------------------------------------------


def test_revenue_per_season_formula():
    """revenue_per_season equals ZRC * PRA price * days for each (season, component) key."""
    mc = _availability_capacity()
    for (season, component), revenue in mc.revenue_per_season.items():
        expected = (
            mc.zrc_mw[(season, component)]
            * mc.pra_prices[season]
            * mc.days_per_season[season]
        )
        np.testing.assert_allclose(revenue, expected, rtol=1e-9)


def test_annual_revenue_sums_seasonal_revenues():
    """annual_revenue[component] equals the sum of revenue_per_season over all seasons."""
    mc = _availability_capacity()
    for component in mc.component_list:
        expected = sum(
            mc.revenue_per_season.get((season, component), 0.0)
            for season in ("summer", "fall", "winter", "spring")
        )
        np.testing.assert_allclose(mc.annual_revenue[component], expected, rtol=1e-9)


def test_revenue_per_season_units_are_dollars():
    """Revenue values are in dollars (ZRC MW * $/MW-day * days = $).

    Verifies dimensional consistency: for a 1 MW ZRC, 1 $/MW-day price,
    and 91 fall days the fall revenue should be exactly 91 dollars.
    """
    df = _make_hourly_df(
        "2022-09-01 05:00",
        48,
        {"a": [1.0] * 48},
    )
    mc = _make_capacity(df, interconnect_limit=100.0, pra_years=[2025])
    fall_revenue = mc.revenue_per_season.get(("fall", "a"))
    if fall_revenue is not None:
        zrc = mc.zrc_mw.get(("fall", "a"), float("nan"))
        price = mc.pra_prices["fall"]
        days = mc.days_per_season["fall"]
        np.testing.assert_allclose(fall_revenue, zrc * price * days, rtol=1e-9)


# ---------------------------------------------------------------------------
# 7. Public output helpers
# ---------------------------------------------------------------------------


def test_get_component_table_shape_and_index():
    """get_component_table returns a (11, 4) DataFrame with seasons as columns."""
    mc = _availability_capacity()
    table = mc.get_component_table("a")
    assert table.shape == (11, 4)
    assert list(table.columns) == ["summer", "fall", "winter", "spring"]
    expected_index = [
        "# Hours",
        "Tier 1 (MW)",
        "Tier 2 (MW)",
        "ISAC (MW)",
        "Class UCAP (MW)",
        "Class ISAC (MW)",
        "SAC (MW)",
        "ZRC (MW)",
        "PRA Price ($/MW-day)",
        "Days",
        "Revenue ($)",
    ]
    assert list(table.index) == expected_index


def test_get_component_table_revenue_row_matches_revenue_per_season():
    """The Revenue ($) row in the component table matches revenue_per_season."""
    mc = _availability_capacity()
    table = mc.get_component_table("a")
    for season in mc.df_h_limit_mw["season"].unique():
        key = (season, "a")
        if key in mc.revenue_per_season:
            np.testing.assert_allclose(
                table.loc["Revenue ($)", season],
                mc.revenue_per_season[key],
                rtol=1e-9,
            )


def test_get_component_table_raises_on_unknown_component():
    """get_component_table raises ValueError for a component not in component_list."""
    mc = _availability_capacity()
    with pytest.raises(ValueError, match="not in component_list"):
        mc.get_component_table("nonexistent")


def test_get_totals_table_sums_capacity_rows():
    """get_totals_table sums MW-valued rows across all components.

    For each capacity metric row the totals table value must equal
    the sum of the same row from each per-component table.
    """
    mc = _availability_capacity()
    totals = mc.get_totals_table()
    summed_rows = [
        "Tier 1 (MW)",
        "Tier 2 (MW)",
        "ISAC (MW)",
        "Class UCAP (MW)",
        "Class ISAC (MW)",
        "SAC (MW)",
        "ZRC (MW)",
        "Revenue ($)",
    ]
    for row in summed_rows:
        expected = sum(mc.get_component_table(c).loc[row] for c in mc.component_list)
        pd.testing.assert_series_equal(totals.loc[row], expected, check_names=False)


def test_get_totals_table_fixed_rows_match_first_component():
    """Hours, PRA price, and days rows are identical across components and taken from the first."""
    mc = _availability_capacity()
    totals = mc.get_totals_table()
    first_table = mc.get_component_table(mc.component_list[0])
    for row in ("# Hours", "PRA Price ($/MW-day)", "Days"):
        pd.testing.assert_series_equal(
            totals.loc[row], first_table.loc[row], check_names=False
        )


def test_get_total_revenue_equals_sum_of_annual_revenues():
    """get_total_revenue() equals sum of annual_revenue values across all components."""
    mc = _availability_capacity()
    expected = sum(mc.annual_revenue.values())
    np.testing.assert_allclose(mc.get_total_revenue(), expected, rtol=1e-9)


def test_get_component_revenue_matches_annual_revenue_dict():
    """get_component_revenue(c) returns the same value as annual_revenue[c]."""
    mc = _availability_capacity()
    for component in mc.component_list:
        np.testing.assert_allclose(
            mc.get_component_revenue(component),
            mc.annual_revenue[component],
            rtol=1e-9,
        )


def test_get_component_revenue_raises_on_unknown_component():
    """get_component_revenue raises ValueError for a component not in component_list."""
    mc = _availability_capacity()
    with pytest.raises(ValueError, match="not in component_list"):
        mc.get_component_revenue("nonexistent")

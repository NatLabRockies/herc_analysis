"""Module for computing MISO capacity from a HERCULES output simulation."""

from pathlib import Path

import numpy as np
import pandas as pd

RA_HOURS_FEATHER_PATH = (
    Path(__file__).parent / "miso_capacity_inputs" / "miso_ra_hours.feather"
)

_VALID_NORTH_SOUTH = {"north", "south"}
_TIER_COLUMNS = ("tier_1", "tier_2", "aaoc")

_DAYS_PER_SEASON_LUT = {
    "summer": 92,
    "fall": 91,
    "winter": 90,
    "spring": 92,
}

_DAYS_PER_SEASON_LUT_LEAP_YEAR = {
    "summer": 92,
    "fall": 91,
    "winter": 91,
    "spring": 92,
}


def get_days_per_season(season: str, year: int) -> int:
    """Get the number of days in a season for a given year."""

    # Determine if the year is a leap year
    is_leap_year = year % 4 == 0
    if is_leap_year:
        return _DAYS_PER_SEASON_LUT_LEAP_YEAR[season]
    else:
        return _DAYS_PER_SEASON_LUT[season]


def compute_battery_availability(
    df: pd.DataFrame,
    component_name: str,
    battery_power_column: str,
    battery_soc_column: str,
    battery_rated_power: float,
    battery_rated_energy: float,
    battery_min_soc: float,
    eta_discharge: float = 0.9,
    return_df_hour: bool = False,
) -> pd.DataFrame:
    """Compute the battery availability for a given dataframe.

    Adds back to df as a new column: {component_name}_availability
    Given a batteries power output and SOC overtime, determine the availablity.
    Note the availability is computed hourly but will upsampled back to the
    original time resolution.

    Args:
        df (pd.DataFrame): DataFrame containing the battery power and SOC columns.
            Must contain the columns:
            - time_utc: timezone-aware UTC datetime
            - battery_power_column: battery power output [kW]
            - battery_soc_column: battery SOC [0-1]

        component_name (str): Name of the component.
        battery_power_column (str): Name of the battery power column.
        battery_soc_column (str): Name of the battery SOC column.
        battery_rated_power (float): Rated power of the battery [kW].
        battery_rated_energy (float): Rated energy of the battery [kWh].
        battery_min_soc (float): Minimum SOC of the battery [0-1].
        eta_discharge (float): Discharge efficiency of the battery [0-1].
        return_df_hour (bool, optional): If True, return the dataframe with the hourly values.
            Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with the battery availability column added.
    """

    # Check that the dataframe contains the required columns
    if "time_utc" not in df.columns:
        raise ValueError("DataFrame must contain a 'time_utc' column.")
    if battery_power_column not in df.columns:
        raise ValueError(f"DataFrame must contain a '{battery_power_column}' column.")
    if battery_soc_column not in df.columns:
        raise ValueError(f"DataFrame must contain a '{battery_soc_column}' column.")

    # Make a copy of the dataframe
    df_hour = df.copy()

    # Floor the time_utc column to the hour
    df_hour["time_utc"] = df_hour["time_utc"].dt.floor("h")

    # Transform the battery power and SOC columns to hourly values
    df_hour["power_hourly"] = (
        df_hour[battery_power_column].groupby(df_hour["time_utc"]).transform("mean")
    )

    # For the capacity calculation, use the SOC at the start of each hour
    # rather than the mean SOC over the hour.  Mean SOC systematically
    # underestimates the energy available "for the next hour" whenever the
    # battery is discharging (mean < start-of-hour value), which would in
    # turn underestimate the output potential.
    df_hour["soc_hour_start"] = (
        df_hour[battery_soc_column].groupby(df_hour["time_utc"]).transform("first")
    )

    # Compute net output
    df_hour["battery_net_output"] = df_hour["power_hourly"].clip(lower=0.0)

    # Compute grid side deliverable energy [kWh] from the SOC at the start
    # of the hour.
    df_hour["grid_side_deliverable_energy"] = (
        (df_hour["soc_hour_start"] - battery_min_soc)
        * battery_rated_energy
        * eta_discharge
    )
    df_hour["grid_side_deliverable_energy"] = df_hour[
        "grid_side_deliverable_energy"
    ].clip(lower=0.0)

    # Compute the output potential power [kW].
    #
    # Note on units: ``grid_side_deliverable_energy`` is in kWh and
    # ``battery_rated_power`` is in kW.  The numerical ``min`` only yields a
    # power because the analysis window is exactly one hour, so kWh and kW
    # share the same number.  Concretely we are computing
    # ``min(rated_power, grid_side_deliverable_energy / 1h)``.  If this
    # function is ever generalized to non-hourly windows, divide the energy
    # by the window length explicitly.
    df_hour["output_potential_power"] = np.minimum(
        df_hour["grid_side_deliverable_energy"], battery_rated_power
    )

    # Compute the availability
    df_hour["battery_availability"] = np.maximum(
        df_hour["output_potential_power"], df_hour["battery_net_output"]
    )

    # Add the battery availability column to the original dataframe
    df[f"{component_name}_availability"] = df_hour["battery_availability"]

    # If return_df_hour is True, return the dataframe with the hourly values
    if return_df_hour:
        return df, df_hour
    else:
        return df


def limit_hourly_availability_contributions(
    df: pd.DataFrame,
    availability_columns: list[str],
    limit: float,
    priority_order: bool = True,
) -> pd.DataFrame:
    """Limit the hourly availability contributions to a maximum of ``limit``.

    For each column in ``availability_columns``, reduce the total power so
    that the average of the per-row sum over each clock hour (in UTC) is at
    most ``limit``.  Time steps within an hour may be much shorter than one
    hour; the cap is applied to the hourly average of the total, not to any
    single sample.  Within an hour, each affected column is scaled
    multiplicatively so the relative shape of its sub-hourly profile is
    preserved.

    Under the default ``priority_order`` behavior, the reduction is drawn
    first from the last column in ``availability_columns``, then the second
    to last, and so on.  Each column can contribute at most its own
    non-negative hourly mean to the reduction (so scaled values stay in
    ``[0, original]``); any remaining excess cascades to the next column.
    When ``priority_order`` is False, the reduction is done pro-rata: every
    column in the hour is multiplied by the same ``limit / hourly_mean``
    factor.

    Args:
        df (pd.DataFrame): DataFrame containing at minimum the columns
            ``time_utc`` (timezone-aware UTC datetime) and every entry of
            ``availability_columns`` (numeric).
        availability_columns (list[str]): List of column names to limit.
        limit (float): Maximum allowed hourly average of the row-wise sum
            over ``availability_columns``.
        priority_order (bool, optional): If True, reduce the contributions
            in reverse list order (last column first).  If False, scale
            every column by the same factor.  Defaults to True.

    Returns:
        pd.DataFrame: A copy of ``df`` with the same shape and column
        names, where the values in ``availability_columns`` have been
        scaled so that the hourly average of their row-wise sum is at most
        ``limit``.

    Raises:
        ValueError: If ``time_utc`` or any entry of ``availability_columns``
            is missing from ``df``, or if ``time_utc`` is not
            timezone-aware.
        TypeError: If ``time_utc`` is not a datetime dtype.
    """
    if "time_utc" not in df.columns:
        raise ValueError("DataFrame must contain a 'time_utc' column.")
    for column in availability_columns:
        if column not in df.columns:
            raise ValueError(f"DataFrame must contain a '{column}' column.")

    time_utc = df["time_utc"]
    if not pd.api.types.is_datetime64_any_dtype(time_utc):
        raise TypeError(
            "Column 'time_utc' must be a datetime dtype (e.g. produced by "
            "pd.to_datetime(..., utc=True))."
        )
    if getattr(time_utc.dt, "tz", None) is None:
        raise ValueError(
            "Column 'time_utc' must be timezone-aware UTC (e.g. produced by "
            "pd.to_datetime(..., utc=True))."
        )

    df_limited = df.copy()

    # Group every row by its hour-beginning UTC bucket so that
    # ``transform("mean")`` broadcasts each hour's mean back to all of its
    # constituent (potentially sub-hourly) rows.
    hour = time_utc.dt.floor("h")
    hourly_mean_total = (
        df[list(availability_columns)].sum(axis=1).groupby(hour).transform("mean")
    )

    if priority_order:
        # Per-row excess of the hourly mean over the limit (constant within
        # each hour).  This is the budget of mean-power we still need to
        # remove from the remaining columns.
        remaining_excess = (hourly_mean_total - limit).clip(lower=0.0)

        for col in reversed(availability_columns):
            hourly_mean_col = df[col].groupby(hour).transform("mean")
            # A column can absorb at most its own non-negative hourly mean
            # without flipping sign.
            available = hourly_mean_col.clip(lower=0.0)
            reduction = np.minimum(remaining_excess, available)

            # scale = 1 - reduction / mean, with the convention that hours
            # whose mean for this column is zero are left untouched.
            ratio = reduction.div(hourly_mean_col.where(hourly_mean_col > 0)).fillna(
                0.0
            )
            scale = 1.0 - ratio
            df_limited[col] = df[col] * scale
            remaining_excess = remaining_excess - reduction
    else:
        # Pro-rata: scale every column by the same hour-specific factor so
        # that the new hourly mean equals ``limit`` (or stays unchanged when
        # already below the limit).
        scale = (limit / hourly_mean_total.where(hourly_mean_total > limit)).fillna(1.0)
        for col in availability_columns:
            df_limited[col] = df[col] * scale

    return df_limited


def compute_capacity_by_tier(
    df: pd.DataFrame,
    availability_column: str,
    north_south: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute mean availability by season, year, and MISO RA-hour tier.

    Each timestamp in ``df`` is associated with a MISO season and RA-hour
    classification via a backward as-of merge against the bundled MISO
    RA-hour reference table (one row per hour, hour-beginning UTC).  The
    most recent reference row whose ``time_utc`` is at or before the
    timestamp wins; this correctly handles ``df`` resolutions finer than
    one hour.

    MISO publishes two independent reliability flags per hour: a seasonal
    RA-hour flag and an annual RA-hour (AAOC) flag.  For each
    ``(season, year)`` combination the following statistics over
    ``availability_column`` are reported:

    - ``tier_1``: mean over hours that are NOT seasonal RA hours.
    - ``tier_2``: mean over seasonal RA hours.  If fewer than 65
      seasonal RA hours are present for the ``(season, year)``
      combination, the values are right-padded up to 65 entries using
      that year's AAOC mean before averaging.
    - ``aaoc``: mean over AAOC hours for the year (season-independent).
    - ``ISAC``: weighted score ``0.2 * tier_1 + 0.8 * tier_2``.
    - ``all``: mean over every classified hour in the
      ``(season, year)`` combination.

    Rows in ``df`` whose ``time_utc`` falls before the first hour of the
    RA-hour reference table are silently dropped because they cannot be
    classified.

    Args:
        df (pd.DataFrame): DataFrame containing at minimum the columns
            ``time_utc`` (timezone-aware UTC datetime) and
            ``availability_column`` (numeric).
        availability_column (str): Name of the numeric column to average
            within each tier.
        north_south (str): MISO sub-region selector. Must be either
            ``"north"`` (Central + North) or ``"south"``.
        verbose (bool, optional): If True, print a status message for
            each ``(season, year)`` combination as it is processed.
            Defaults to False.

    Returns:
        pd.DataFrame: Wide DataFrame whose columns are ``(season, year)``
        tuples and whose index is ``["tier_1", "tier_2", "aaoc", "ISAC",
        "all"]``.  Each column holds the statistics described above for
        that ``(season, year)`` combination.

    Raises:
        ValueError: If required columns are missing, ``time_utc`` is not
            timezone-aware, or ``north_south`` is invalid.
        TypeError: If ``time_utc`` is not a datetime dtype.
        KeyError: If the ``(season, year)`` combination has no AAOC
            hours available for that year.
    """
    if "time_utc" not in df.columns:
        raise ValueError("DataFrame must contain a 'time_utc' column.")
    if availability_column not in df.columns:
        raise ValueError(f"DataFrame must contain a '{availability_column}' column.")
    if north_south not in _VALID_NORTH_SOUTH:
        raise ValueError(
            f"north_south must be one of {sorted(_VALID_NORTH_SOUTH)}, "
            f"got {north_south!r}."
        )

    time_utc = df["time_utc"]
    if not pd.api.types.is_datetime64_any_dtype(time_utc):
        raise TypeError(
            "Column 'time_utc' must be a datetime dtype (e.g. produced by "
            "pd.to_datetime(..., utc=True))."
        )
    if getattr(time_utc.dt, "tz", None) is None:
        raise ValueError(
            "Column 'time_utc' must be timezone-aware UTC (e.g. produced by "
            "pd.to_datetime(..., utc=True))."
        )

    ra_column = f"ra_{north_south}"
    aaoc_column = f"aaoc_{north_south}"

    df_ra = pd.read_feather(RA_HOURS_FEATHER_PATH)

    # merge_asof requires both inputs to be sorted on the join key.
    df_left = (
        df[["time_utc", availability_column]]
        .sort_values("time_utc")
        .reset_index(drop=True)
    )
    df_right = (
        df_ra[["time_utc", "season", ra_column, aaoc_column]]
        .sort_values("time_utc")
        .reset_index(drop=True)
    )

    df_merge = pd.merge_asof(
        df_left,
        df_right,
        on="time_utc",
        direction="backward",
    )

    # Drop rows that precede the start of the reference table; they have
    # no season/RA classification and would otherwise pollute the means.
    df_merge = df_merge.dropna(subset=["season"]).copy()
    df_merge["year"] = df_merge["time_utc"].dt.year

    # First compute the AAOC means.
    df_aaoc = df_merge[df_merge[aaoc_column]]
    df_aaoc_means = df_aaoc.groupby(["year"])[availability_column].mean().rename("aaoc")

    # Convert to a dictionary with keys "year"
    aaoc_means = df_aaoc_means.to_dict()

    result_dict = {}

    # Now loop over season, year combinations in df_merge
    for season, year in (
        df_merge[["season", "year"]].drop_duplicates().itertuples(index=False)
    ):
        if verbose:
            print(f"Processing season {season}, year {year}")
        df_subset = df_merge[df_merge["season"] == season]
        df_subset = df_subset[df_subset["year"] == year]

        # First get the Tier 2 dataframe
        df_tier_2_values = df_subset[df_subset[ra_column]][availability_column].values

        # If df_tier_2 has < 65 rows, pad with the aaoc mean for that year until it has 65 rows
        if len(df_tier_2_values) < 65:
            df_tier_2_values = np.pad(
                df_tier_2_values,
                (0, 65 - len(df_tier_2_values)),
                mode="constant",
                constant_values=aaoc_means[year],
            )

        # Now get the mean value for Tier 2
        tier_2_mean = np.mean(df_tier_2_values)

        # Now get the tier 1 mean (no padding required)
        tier_1_mean = np.mean(
            df_subset[~df_subset[ra_column]][availability_column].values
        )

        # Compute the ISAC value via weighted sum
        isac_value = 0.2 * tier_1_mean + 0.8 * tier_2_mean

        # Finally compute the mean over all data
        all_mean = np.mean(df_subset[availability_column].values)

        # Add the results to the result_dict
        result_dict[season, year] = {
            "tier_1": tier_1_mean,
            "tier_2": tier_2_mean,
            "aaoc": aaoc_means[year],
            "ISAC": isac_value,
            "all": all_mean,
        }

    return pd.DataFrame(result_dict)


if __name__ == "__main__":
    ra_hours = pd.read_feather(RA_HOURS_FEATHER_PATH)
    print(ra_hours.head())

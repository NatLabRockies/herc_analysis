"""Module for computing MISO capacity from a HERCULES output simulation."""

import calendar
from pathlib import Path

import numpy as np
import pandas as pd

RA_HOURS_FEATHER_PATH = (
    Path(__file__).parent / "miso_capacity_inputs" / "miso_ra_hours.feather"
)

_VALID_NORTH_SOUTH = {"north", "south"}


def _coerce_to_bool_mask(series: pd.Series, column_name: str) -> pd.Series:
    """Coerce an RA-hour flag column to a clean boolean mask.

    Accepts either a true ``bool`` dtype column or a numeric column whose
    values are all in ``{0, 1}`` (NaN is treated as ``False``).  Any
    other dtype, or any non-{0, 1} numeric value, raises an error.

    Args:
        series (pd.Series): Column of RA-hour flags to coerce.
        column_name (str): Original column name, used only for error
            messages.

    Returns:
        pd.Series: Boolean Series aligned to ``series``.

    Raises:
        TypeError: If ``series`` is neither bool nor numeric.
        ValueError: If ``series`` is numeric but contains values other
            than 0 or 1 (NaN is allowed and is mapped to ``False``).
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        unique_values = set(pd.unique(series.dropna()))
        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Column '{column_name}' must contain only 0/1 or "
                f"True/False values; got {sorted(unique_values)}."
            )
        return series.fillna(0).astype(bool)
    raise TypeError(
        f"Column '{column_name}' must be a bool or 0/1 numeric column, "
        f"got dtype {series.dtype}."
    )


# MISO season -> day count.  Seasons follow the meteorological convention
# used internally by HERCULES: spring = Mar-May, summer = Jun-Aug,
# fall = Sep-Nov, winter = Dec(year-1) + Jan-Feb(year), so the leap day
# (Feb 29 of ``year``) lengthens the winter that *ends* in ``year``.
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
    """Get the number of days in a MISO season for a given year.

    Uses the Gregorian leap-year rule: a year is a leap year iff it is
    divisible by 4, except century years which must also be divisible by
    400.  In a leap year ``winter`` (Dec(year-1) + Jan-Feb(year)) gains
    the extra day; the other seasons are unchanged.

    Args:
        season (str): MISO season name. Must be one of ``"spring"``,
            ``"summer"``, ``"fall"``, ``"winter"``.
        year (int): Calendar year used to decide whether the leap-year
            day count applies.

    Returns:
        int: Number of days in ``season`` for ``year``.

    Raises:
        KeyError: If ``season`` is not one of the four supported values.
    """
    if calendar.isleap(year):
        return _DAYS_PER_SEASON_LUT_LEAP_YEAR[season]
    return _DAYS_PER_SEASON_LUT[season]


def compute_battery_availability(
    df: pd.DataFrame,
    component_name: str,
    battery_power_column: str,
    battery_soc_column: str,
    battery_rated_power: float,
    battery_rated_energy: float,
    battery_min_soc: float,
    eta_discharge: float,
    return_df_hour: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the battery availability for a given dataframe.

    Returns a copy of ``df`` with a new column ``{component_name}_availability``.
    Given a battery's power output and SOC over time, determine its
    availability for the next hour.  The availability is computed hourly
    and then broadcast back ("upsampled") to the input's original
    sub-hourly resolution by repeating each hour's value across all of
    its constituent rows.

    Args:
        df (pd.DataFrame): DataFrame containing the battery power and SOC
            columns.  Must contain the columns:

            - ``time_utc``: timezone-aware UTC datetime
            - ``battery_power_column``: battery power output [kW]
            - ``battery_soc_column``: battery SOC [0-1]

        component_name (str): Name of the component.
        battery_power_column (str): Name of the battery power column.
        battery_soc_column (str): Name of the battery SOC column.
        battery_rated_power (float): Rated power of the battery [kW].
        battery_rated_energy (float): Rated energy of the battery [kWh].
        battery_min_soc (float): Minimum SOC of the battery [0-1].
        eta_discharge (float): Discharge efficiency of the battery [0-1].
        return_df_hour (bool, optional): If True, also return the
            intermediate hourly working DataFrame (which contains the
            per-hour SOC, energy, and availability columns).  Defaults
            to False.

    Returns:
        pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]: A copy of
        ``df`` with the ``{component_name}_availability`` column added.
        If ``return_df_hour`` is True, a tuple ``(df_return, df_hour)``
        is returned instead, where ``df_hour`` is the hourly
        intermediate DataFrame.
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
    df_return = df.copy()
    df_return[f"{component_name}_availability"] = df_hour["battery_availability"]

    # If return_df_hour is True, return the dataframe with the hourly values
    if return_df_hour:
        return df_return, df_hour
    else:
        return df_return


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


def compute_isac_dict(
    df: pd.DataFrame,
    availability_column: str,
    north_south: str,
    verbose: bool = False,
) -> dict[str, dict[str, float]]:
    """Compute MISO ISAC tier statistics for each season.

    The input ``df`` is first floor-aggregated to hour-beginning UTC by
    averaging ``availability_column`` within each clock hour, and the
    result is inner-joined to the bundled MISO RA-hour reference table
    on ``time_utc``.  Hours with no entry in the reference table are
    dropped, as are years with fewer than ``24 * 275`` classified hours
    (a "near-full-year" filter).

    MISO publishes two independent reliability flags per hour: a
    seasonal RA-hour flag (``ra_<region>``) and an annual RA-hour /
    AAOC flag (``aaoc_<region>``).  For each season the following
    statistics over ``availability_column`` are reported, pooled across
    every kept year:

    - ``tier_1``: mean over hours that are NOT seasonal RA hours.
    - ``tier_2``: mean over seasonal RA hours, pooled across years.  For
      any year with fewer than 65 seasonal RA hours, that year's
      contribution is right-padded up to 65 entries using that year's
      AAOC mean before being concatenated.
    - ``ISAC``: weighted score ``0.2 * tier_1 + 0.8 * tier_2``.
    - ``aaoc``: mean over every AAOC hour in the kept years
      (season-independent; identical across seasons).
    - ``all``: mean over every classified hour in the season (across
      kept years).

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
        dict[str, dict[str, float]]: Nested mapping from season name to
        a sub-dict with keys ``"tier_1"``, ``"tier_2"``, ``"ISAC"``,
        ``"aaoc"``, and ``"all"``, each holding the statistics described
        above.

    Raises:
        ValueError: If required columns are missing or ``north_south``
            is invalid, or if the ``ra_<region>`` / ``aaoc_<region>``
            reference columns contain non-{0, 1} values.
        TypeError: If the ``ra_<region>`` / ``aaoc_<region>`` reference
            columns are neither bool nor numeric.
        KeyError: If a season/year combination needs AAOC padding but
            that year has no AAOC hours available.
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

    # Compute the hourly averages
    df_hourly = (
        df.groupby(df["time_utc"].dt.floor("h"))[availability_column]
        .mean()
        .reset_index()
    )

    # Load the RA-hour reference table
    df_ra = pd.read_feather(RA_HOURS_FEATHER_PATH)

    # Determine the RA and AAOC columns
    ra_column = f"ra_{north_south}"
    aaoc_column = f"aaoc_{north_south}"

    df_merge = pd.merge(df_hourly, df_ra, on="time_utc", how="inner")

    # Coerce flag columns to clean booleans so callers can supply either
    # bool or 0/1 numeric reference tables without surprises.
    df_merge[ra_column] = _coerce_to_bool_mask(df_merge[ra_column], ra_column)
    df_merge[aaoc_column] = _coerce_to_bool_mask(df_merge[aaoc_column], aaoc_column)

    df_merge["year"] = df_merge["time_utc"].dt.year

    # Per-year AAOC means, used both for tier_2 padding and for the
    # season-independent "aaoc" output entry.
    aaoc_means = (
        df_merge[df_merge[aaoc_column]]
        .groupby("year")[availability_column]
        .mean()
        .to_dict()
    )

    seasons = df_merge["season"].unique()
    years = [
        year
        for year in df_merge["year"].unique()
        if len(df_merge[df_merge["year"] == year]) > 24 * 275
    ]

    # Season-independent AAOC mean: average over every AAOC hour in the
    # kept years.  Identical across seasons, but emitted per season for
    # convenience.
    aaoc_mask_kept = df_merge[aaoc_column] & df_merge["year"].isin(years)
    aaoc_overall_mean = float(df_merge.loc[aaoc_mask_kept, availability_column].mean())

    result_dict: dict[str, dict[str, float]] = {}

    for season in seasons:
        df_season = df_merge[df_merge["season"] == season]
        tier_1_values = np.array([])
        tier_2_values = np.array([])

        for year in years:
            df_subset = df_season[df_season["year"] == year]

            tier_2_year = df_subset.loc[
                df_subset[ra_column], availability_column
            ].values
            if len(tier_2_year) < 65:
                tier_2_year = np.pad(
                    tier_2_year,
                    (0, 65 - len(tier_2_year)),
                    mode="constant",
                    constant_values=aaoc_means[year],
                )

            tier_1_year = df_subset.loc[
                ~df_subset[ra_column], availability_column
            ].values

            tier_1_values = np.concatenate([tier_1_values, tier_1_year])
            tier_2_values = np.concatenate([tier_2_values, tier_2_year])

            if verbose:
                print(
                    f"[compute_isac_dict] season={season} year={year}: "
                    f"tier_1_n={len(tier_1_year)} tier_2_n={len(tier_2_year)}"
                )

        tier_1_mean = (
            float(np.mean(tier_1_values)) if tier_1_values.size else float("nan")
        )
        tier_2_mean = (
            float(np.mean(tier_2_values)) if tier_2_values.size else float("nan")
        )
        isac_value = 0.2 * tier_1_mean + 0.8 * tier_2_mean

        # Per-season "all" mean: every classified hour in this season,
        # across kept years (matches the docstring's wording).
        season_mask = df_season["year"].isin(years)
        all_mean = float(df_season.loc[season_mask, availability_column].mean())

        result_dict[season] = {
            "tier_1": tier_1_mean,
            "tier_2": tier_2_mean,
            "ISAC": isac_value,
            "aaoc": aaoc_overall_mean,
            "all": all_mean,
        }

    return result_dict


if __name__ == "__main__":
    ra_hours = pd.read_feather(RA_HOURS_FEATHER_PATH)
    print(ra_hours.head())

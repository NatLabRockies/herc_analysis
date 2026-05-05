"""Utilities for herc_analysis."""

from functools import cache

import numpy as np
import pandas as pd
import polars as pl
from hercules.utilities import hercules_float_type
from timezonefinder import TimezoneFinder

_VALID_INTERPOLATION_METHODS = {
    "averaged_to_instantaneous",
    "instantaneous_to_instantaneous",
}


@cache
def _get_timezone_finder():
    """Return a process-wide cached :class:`TimezoneFinder` instance.

    Constructing a :class:`TimezoneFinder` loads the bundled timezone
    polygon dataset, which is moderately expensive.  Caching the instance
    keeps repeated lookups cheap.

    Returns:
        TimezoneFinder: A cached :class:`TimezoneFinder` instance.
    """
    return TimezoneFinder()


def add_local_time(df, latitude, longitude):
    """Append a ``time_local`` column derived from ``time_utc``.

    The IANA timezone is determined from the supplied geographic
    coordinates using :mod:`timezonefinder`, and ``time_utc`` is converted
    into that timezone to produce ``time_local``.  The resulting column is
    timezone-aware (call ``.dt.tz_localize(None)`` afterwards if a naive
    wall-clock representation is desired).

    Args:
        df (pd.DataFrame): DataFrame containing a ``time_utc`` column with
            timezone-aware UTC timestamps (e.g. as produced by
            ``pd.to_datetime(..., utc=True)``).
        latitude (float): Latitude in decimal degrees.
        longitude (float): Longitude in decimal degrees.

    Returns:
        pd.DataFrame: A copy of *df* with an additional ``time_local``
        column holding timezone-aware timestamps in the local timezone of
        the supplied coordinates.

    Raises:
        ValueError: If ``time_utc`` is missing, not timezone-aware, or if
            no timezone could be resolved for the supplied coordinates.
        TypeError: If ``time_utc`` is not a datetime dtype.
    """
    if "time_utc" not in df.columns:
        raise ValueError("DataFrame must contain a 'time_utc' column.")

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

    tz_name = _get_timezone_finder().timezone_at(lat=latitude, lng=longitude)
    if tz_name is None:
        raise ValueError(
            f"Could not determine timezone for coordinates "
            f"(lat={latitude}, lon={longitude})."
        )

    df = df.copy()
    df["time_local"] = time_utc.dt.tz_convert(tz_name)
    return df


def _compute_interval_midpoints(time_values):
    """Compute the midpoints of consecutive time intervals.

    For start-of-period timestamps, each value is best represented at the
    center of its interval.  The last interval width is assumed equal to the
    preceding one.

    Args:
        time_values (np.ndarray): Sorted array of start-of-period timestamps.

    Returns:
        np.ndarray: Array of interval midpoints, same length as *time_values*.
    """
    # Allow the edge case of a single time value by returning the time value itself
    if len(time_values) < 2:
        return time_values
    # Compute midpoints
    midpoints = np.empty_like(time_values, dtype=np.float64)
    midpoints[:-1] = (time_values[:-1] + time_values[1:]) / 2.0
    midpoints[-1] = (
        time_values[-1] + (time_values[-1] - time_values[-2]) / 2.0
    )  # Last interval is equal to the previous one
    return midpoints


def interpolate_df(df, new_time, interpolation_method):
    """Interpolate DataFrame values to match new time axis.

    The ``interpolation_method`` parameter controls how numeric columns are
    resampled onto ``new_time``:

    - ``"averaged_to_instantaneous"``: Input values are period averages whose
      timestamps mark the **start** of each period.  Each value is assigned to
      the midpoint of its interval and then linearly interpolated.  Use for
      wind speed, solar irradiance, and similar time-averaged signals.
    - ``"instantaneous_to_instantaneous"``: Input values already represent
      instantaneous measurements.  Standard linear interpolation is performed
      directly on the original timestamps with no midpoint shift.

    Datetime columns (e.g. ``time_utc``) are always linearly interpolated on
    the raw timestamps regardless of the chosen method, because they map
    simulation time to wall-clock time directly.

    Note:
        A dedicated zero-order-hold (ZOH) mode is intentionally not provided.
        If you need step/piecewise-constant behaviour (e.g. LMP prices that
        should be held constant across each reporting interval), pre-process
        the input DataFrame to include an extra row at the end of each
        interval carrying the same value, and then call this function with
        ``"instantaneous_to_instantaneous"``.  Linear interpolation between
        each pair of identical endpoints reproduces the ZOH shape.  See
        ``hercules.grid.grid_utilities.generate_locational_marginal_price_dataframe_from_gridstatus``
        for an example of this endpoint-insertion pattern.

    Args:
        df (pd.DataFrame): DataFrame with 'time' column and data columns.
        new_time (array-like): New time points for interpolation.
        interpolation_method (str): One of ``"averaged_to_instantaneous"`` or
            ``"instantaneous_to_instantaneous"``.

    Returns:
        pd.DataFrame: DataFrame with new time axis and interpolated data columns.

    """
    if interpolation_method not in _VALID_INTERPOLATION_METHODS:
        raise ValueError(
            f"Unknown interpolation_method '{interpolation_method}'. "
            f"Must be one of {sorted(_VALID_INTERPOLATION_METHODS)}."
        )

    new_time = np.asarray(new_time)

    # Separate datetime and non-datetime columns for different processing
    datetime_cols = []
    numeric_cols = []

    for col in df.columns:
        if col != "time":
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            else:
                numeric_cols.append(col)

    # Sort by "time" once up front so that np.interp (which requires
    # strictly-increasing x-coordinates) sees monotonic input for every
    # column.  Applying the sort in one place keeps numeric and datetime
    # columns consistently ordered.
    df_pl = pl.from_pandas(df).sort("time")
    result_pl = pl.DataFrame({"time": new_time})

    time_values = df_pl["time"].to_numpy()

    if interpolation_method == "averaged_to_instantaneous":
        x_coords = _compute_interval_midpoints(time_values)
    else:
        x_coords = time_values

    for col in numeric_cols:
        col_values = df_pl[col].to_numpy()
        interpolated_values = np.interp(new_time, x_coords, col_values).astype(
            hercules_float_type
        )
        result_pl = result_pl.with_columns(pl.lit(interpolated_values).alias(col))

    # Process datetime columns (use the same sorted frame as numeric cols)
    for col in datetime_cols:
        datetime_values = df_pl[col].to_pandas().astype("int64").values / 10**9

        # Interpolate timestamps (datetime precision doesn't need float32 constraint)
        interpolated_timestamps = np.interp(new_time, time_values, datetime_values)

        # Convert back to datetime and add to result
        interpolated_datetimes = pd.to_datetime(
            interpolated_timestamps, unit="s", utc=True
        )
        result_pl = result_pl.with_columns(pl.Series(col, interpolated_datetimes))

    # Convert back to pandas DataFrame
    return result_pl.to_pandas()

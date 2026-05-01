"""Utilities for herc_analysis."""

import numpy as np
import pandas as pd
import polars as pl
from hercules.utilities import hercules_float_type

_VALID_INTERPOLATION_METHODS = {
    "averaged_to_instantaneous",
    "instantaneous_to_instantaneous",
}


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

"""Module for computing MISO capacity from a HERCULES output simulation."""

import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_INPUTS_DIR = Path(__file__).parent / "miso_capacity_inputs"
RA_HOURS_CSV_PATH = _INPUTS_DIR / "miso_ra_hours.csv"
PRA_PRICES_CSV_PATH = _INPUTS_DIR / "pra_prices_usd_per_mw_day.csv"
RESOURCE_CLASS_UCAP_CSV_PATH = _INPUTS_DIR / "resource_class_ucap_mw.csv"
RESOURCE_CLASS_ISAC_CSV_PATH = _INPUTS_DIR / "resource_class_isac_mw.csv"
DAYS_PER_SEASON_CSV_PATH = _INPUTS_DIR / "days_per_season.csv"
DAYS_PER_SEASON_LEAP_YEAR_CSV_PATH = _INPUTS_DIR / "days_per_season_leap_year.csv"
ZONE_SUBREGION_CSV_PATH = _INPUTS_DIR / "zone_subregion.csv"

_SEASONS = ("summer", "fall", "winter", "spring")
_TIER_2_PAD_TARGET = 65
# Drop a planning year (Sept Y -> Aug Y+1) when its total classified hour
# count is below this threshold.  ``8760 - 24`` allows up to 24 missing
# hours per PY (e.g. minor data drops or DST edge cases).
_LOW_HOUR_PLANNING_YEAR_THRESHOLD_HOURS = 8760 - 24


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


def _time_to_planning_year(timestamps: pd.Series) -> pd.Series:
    """Map UTC timestamps to MISO planning-year codes (e.g. ``2223``).

    The planning year runs Sept Y -> Aug Y+1.  In MISO's reference table
    the boundary is Sept 1 EST = Sept 1 05:00 UTC; for the purposes of
    this helper we use the simpler UTC month boundary, which is correct
    everywhere except a 5-hour window on Sept 1.  The helper is intended
    only for the standalone coverage plot, where exact-hour fidelity is
    not required.
    """
    years = timestamps.dt.year
    months = timestamps.dt.month
    start = years.where(months >= 9, years - 1)
    return (start % 100) * 100 + (start + 1) % 100


def plot_planning_year_coverage(
    df: pd.DataFrame,
    save_path: Path | None = None,
) -> "plt.Figure":
    """Plot per-column timeseries colored by planning-year completeness.

    Standalone helper that lets a user inspect a raw timeseries and see
    which MISO planning years (Sept Y -> Aug Y+1) are complete enough to
    survive :class:`MisoCapacity`'s :attr:`remove_low_hour_planning_years`
    filter.  Hours belonging to a complete PY (>=
    :data:`_LOW_HOUR_PLANNING_YEAR_THRESHOLD_HOURS` rows in ``df``) are
    drawn in black; hours in incomplete PYs are drawn in pale red, since
    they would be dropped if ``df`` were fed to :class:`MisoCapacity`
    with the default settings.  An ``<--->`` arrow spanning Sept 1 ->
    Aug 31 is drawn above each complete PY.

    Args:
        df (pd.DataFrame): Tidy frame with a tz-aware UTC ``time_utc``
            column and at least one numeric data column.
        save_path (Path | None): If provided, save the figure here.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    if "time_utc" not in df.columns:
        raise ValueError("DataFrame must contain a 'time_utc' column.")
    if df["time_utc"].dt.tz is None or str(df["time_utc"].dt.tz) != "UTC":
        raise ValueError("Time_utc column must be in UTC timezone.")
    data_cols = [c for c in df.columns if c != "time_utc"]
    if not data_cols:
        raise ValueError("DataFrame must contain at least one non-time_utc column.")

    df_plot = df.copy()
    df_plot["planning_year"] = _time_to_planning_year(df_plot["time_utc"])
    py_counts = df_plot.groupby("planning_year").size()
    complete_pys = sorted(
        int(py)
        for py, n in py_counts.items()
        if n >= _LOW_HOUR_PLANNING_YEAR_THRESHOLD_HOURS
    )
    is_complete = df_plot["planning_year"].isin(complete_pys)

    fig, axes = plt.subplots(
        len(data_cols), 1, figsize=(11, 2.5 * len(data_cols)), sharex=True
    )
    if len(data_cols) == 1:
        axes = [axes]

    times = df_plot["time_utc"]
    for ax, col in zip(axes, data_cols, strict=True):
        ax.scatter(
            times[~is_complete],
            df_plot.loc[~is_complete, col],
            s=4,
            color="#f4b6b6",
            label="incomplete PY (will be dropped)",
        )
        ax.scatter(
            times[is_complete],
            df_plot.loc[is_complete, col],
            s=4,
            color="black",
            label="complete PY",
        )
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

        # Arrow + label above the data for each complete PY.
        y_top = ax.get_ylim()[1]
        for py in complete_pys:
            start_year = 2000 + py // 100
            start = pd.Timestamp(f"{start_year}-09-01", tz="UTC")
            end = pd.Timestamp(f"{start_year + 1}-09-01", tz="UTC")
            ax.annotate(
                "",
                xy=(end, y_top),
                xytext=(start, y_top),
                arrowprops={"arrowstyle": "<->", "color": "tab:blue", "lw": 1.5},
                annotation_clip=False,
            )
            mid = start + (end - start) / 2
            ax.text(
                mid,
                y_top,
                f"PY {py}",
                ha="center",
                va="bottom",
                color="tab:blue",
                fontsize=9,
            )

    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time_utc")
    fig.suptitle("Planning-year coverage")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    return fig


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
        ``df`` with the ``{component_name}_availability`` column added
        (units match the input power column, typically kW).
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


class MisoCapacity:
    def __init__(
        self,
        component_list: list[str],
        class_list: list[str],
        df: pd.DataFrame,
        zone: int,
        interconnect_limit: float,
        pra_years: int | list[int] | None = None,
        priority_order: list[str] | dict[str, list[str]] | None = None,
        remove_low_hour_planning_years: bool = True,
        verbose: bool = False,
    ):
        """Initialize a MisoCapacity analysis for the given components.

        Args:
            component_list (list[str]): Column names in ``df`` representing
                each generation component (e.g. battery, wind).
            class_list (list[str]): MISO resource class for each component
                (same order as ``component_list``).  Must appear in the
                bundled UCAP/ISAC CSVs.
            df (pd.DataFrame): Simulation output containing ``time_utc``
                (timezone-aware UTC) and one column per component.  Sub-hourly
                data are averaged to hourly resolution internally.
            zone (int): MISO zone number used to look up the subregion and
                PRA clearing prices.
            interconnect_limit (float): Maximum interconnect capacity in kW.
                Stored internally as MW (``self.interconnect_limit_mw``).
            pra_years (int | list[int] | None): PRA auction year(s) used for
                clearing prices.  Accepts a single int, a list of ints (prices
                are averaged), or ``None`` / empty list (defaults to the most
                recent year in the bundled CSV).
            priority_order (list[str] | dict[str, list[str]] | None):
                Determines how interconnect headroom is allocated when the
                sum of component outputs exceeds ``interconnect_limit``.
                ``None`` applies pro-rata scaling; a list applies the same
                priority every season; a dict keyed by season applies
                per-season priorities.
            remove_low_hour_planning_years (bool): When ``True`` (default),
                any planning year (Sept Y -> Aug Y+1) with fewer than
                ``_LOW_HOUR_PLANNING_YEAR_THRESHOLD_HOURS`` classified
                hours is excluded *in its entirety* (all four seasons
                together) from the availability metrics.  Set to
                ``False`` in unit tests that use short windows.
            verbose (bool): When ``True``, log diagnostic messages (e.g.
                which planning years were dropped).
        """
        # Save the verbose flag
        self.verbose = verbose

        # Convert kW input to MW for all internal calculations
        self.interconnect_limit_mw = interconnect_limit / 1000.0

        # When True (default), any planning year in ``df_h_limit_mw``
        # with fewer than ``_LOW_HOUR_PLANNING_YEAR_THRESHOLD_HOURS``
        # classified hours is dropped *whole* (all four seasons together)
        # before the per-component availability metrics are computed.
        # Set to False for unit tests on small windows.
        self.remove_low_hour_planning_years = remove_low_hour_planning_years

        # Check that component_list and class_list are lists of strings
        if not isinstance(component_list, list) or not all(
            isinstance(c, str) for c in component_list
        ):
            raise ValueError("component_list must be a list of strings.")
        if not isinstance(class_list, list) or not all(
            isinstance(c, str) for c in class_list
        ):
            raise ValueError("class_list must be a list of strings.")

        # Check that the component_list and class_list are the same length
        if len(component_list) != len(class_list):
            raise ValueError("component_list and class_list must be the same length.")

        # Get a list of available classes
        self.available_classes = self._get_available_classes()

        # Check that the class_list is a subset of the available_classes
        if not all(c in self.available_classes for c in class_list):
            raise ValueError(
                f"One or more classes in {class_list} not found in available classes: {self.available_classes}"
            )

        # Save the component_list and class_list
        self.component_list = component_list
        self.class_list = class_list
        self.n_components = len(component_list)

        # Check that the dataframe contains at least two columns and a time_utc column
        if len(df.columns) < 2:
            raise ValueError("DataFrame must contain at least two columns.")
        if "time_utc" not in df.columns:
            raise ValueError("DataFrame must contain a 'time_utc' column.")
        # Check that the time_utc is UTC.  ``dt.tz`` is a ``tzinfo``
        # object (e.g. ``datetime.timezone.utc``), not the string
        # ``"UTC"``, so compare via its string representation.
        if df["time_utc"].dt.tz is None or str(df["time_utc"].dt.tz) != "UTC":
            raise ValueError("Time_utc column must be in UTC timezone.")
        # Check that the dataframe has at least two rows
        if len(df) < 2:
            raise ValueError("DataFrame must contain at least two rows.")

        # Check the there is a column for each component in the component_list
        for component in component_list:
            if component not in df.columns:
                raise ValueError(f"DataFrame must contain a '{component}' column.")

        # Save the zone
        self.zone = zone

        # Get the subregion from the zone
        self.subregion = self._get_subregion()

        # Convert component columns from kW to MW; store for reference and downstream use
        df_mw = df.copy()
        df_mw[self.component_list] = df_mw[self.component_list] / 1000.0
        self.df_mw = df_mw

        # Process the dataframe (uses self.subregion to pick RA / AAOC cols)
        self.df_h_mw = self._process_to_hourly(self.df_mw)

        # Process the priority order

        # Check the priority is either None, a list of strings, or a dict of lists of strings
        if (
            priority_order is not None
            and not isinstance(priority_order, list)
            and not isinstance(priority_order, dict)
        ):
            raise ValueError(
                f"Priority order must be None, a list of strings, or a dict of lists of strings: {type(priority_order)}"
            )

        # If None can save directly to self.priority_order
        if priority_order is None:
            self.priority_order = priority_order
        # Else if, priority_order is a list of strings, confirm same length as component_list
        # and elements overlap with component_list.  Then save each list as the value of
        # a dictionary whose keys are the four seasons.
        if isinstance(priority_order, list):
            if len(priority_order) != len(component_list):
                raise ValueError(
                    f"Priority order list must be the same length as component_list: {len(component_list)}"
                )
            if not all(component in component_list for component in priority_order):
                raise ValueError(
                    f"One or more components in {priority_order} not found in component_list: {component_list}"
                )
            self.priority_order = dict.fromkeys(_SEASONS, priority_order)
        # Else if, priority_order is a dict first confirm the keys are the four seasons
        # and then confirm the values are lists of strings that overlap with component_list.
        elif isinstance(priority_order, dict):
            if not all(season in _SEASONS for season in priority_order.keys()):
                raise ValueError(
                    f"One or more seasons in {priority_order.keys()} not found in _SEASONS: {_SEASONS}"
                )
            if not all(
                all(component in component_list for component in components)
                for components in priority_order.values()
            ):
                raise ValueError(
                    f"One or more components in {priority_order.values()} not found in component_list: {component_list}"
                )
            self.priority_order = priority_order

        # Limit the component availability contributions to the interconnect limit according to the priority order
        self.df_h_limit_mw = self._limit_hourly_availability_contributions()

        # Compute per-component availability metrics off the limit-capped
        # hourly frame.  Order matters:
        #   1. Capture per-(planning_year, season) hour counts from the
        #      original ``df_h_limit_mw`` (so the diagnostic survives any
        #      later drops).
        #   2. Optionally drop sparsely-covered planning years (whole-PY).
        #   3. Compute per-PY, per-component AAOC means used for the
        #      tier-2 padding step.
        #   4. Compute the tier-1 / tier-2 / all-hours / ISAC dicts.
        self.hours_per_planning_year_season = self._compute_hour_counts()
        self.df_h_limit_mw = self._drop_low_hour_planning_years()
        self.aaoc_per_planning_year_mw = self._compute_aaoc_per_planning_year()
        (
            self.tier_1_availability_mw,
            self.tier_2_availability_mw,
            self.all_hours_availability_mw,
            self.isac_mw,
        ) = self._compute_tier_availabilities()

        # Load the class-level UCAP and ISAC (MW, from MISO reference CSVs)
        self.class_ucap_mw = self._load_class_ucap_mw()
        self.class_isac_mw = self._load_class_isac_mw()

        # Compute class-level UCAP / ISAC conversion factor
        self.class_ucap_isac_conversion = self._compute_class_ucap_isac_conversion()

        # Compute component-level Seasonal Accredited Capacity (MW)
        self.sac_mw = self._compute_sac_mw()

        # For now, assume zrc (Zonal Resource Credits) simply equals sac
        self.zrc_mw = self.sac_mw

        # Get a list of available pra price years
        self.available_pra_years = self._get_available_pra_years()

        # If pra_years is None or an empty list, use most recent value in self.available_pra_years
        if pra_years is None or not pra_years:
            self.pra_years = [self.available_pra_years[-1]]
        # Else if, pra_years is an int, confirm in self.available_pra_years and save it as a list
        elif isinstance(pra_years, int):
            if pra_years not in self.available_pra_years:
                raise ValueError(
                    f"PRA year {pra_years} not found in available years: {self.available_pra_years}"
                )
            self.pra_years = [pra_years]
        # Else if, pra_years is a list of ints, confirm all are in self.available_pra_years and save it as a list
        elif isinstance(pra_years, list):
            if not all(year in self.available_pra_years for year in pra_years):
                raise ValueError(
                    f"One or more PRA years in {pra_years} not found in available years: {self.available_pra_years}"
                )
            self.pra_years = pra_years

        if len(self.pra_years) > 1:
            print(f"Using average PRA prices for multiple years: {self.pra_years}")
        else:
            print(f"Using PRA prices for year: {self.pra_years[0]}")

        # Save the PRA prices per season
        self.pra_prices = self._get_pra_prices()

        # Save the days per season
        self.days_per_season = self._get_days_per_season()

        # Compute the revenue per season
        self.revenue_per_season = self._compute_revenue_per_season()

        # Compute the annual revenue
        self.annual_revenue = self._compute_annual_revenue()

    def _get_available_classes(self) -> list[str]:
        """Get available resource classes from the bundled UCAP CSV.

        Reads :data:`RESOURCE_CLASS_UCAP_CSV_PATH` and returns sorted
        unique values from the ``resource_class`` column.

        Returns:
            list[str]: Sorted list of available resource class names.
        """
        ucap_df = pd.read_csv(RESOURCE_CLASS_UCAP_CSV_PATH)
        return sorted(
            ucap_df["resource_class"].dropna().astype(str).str.strip().unique().tolist()
        )

    def _get_subregion(self) -> str:
        """Get the MISO subregion name for :attr:`zone`.

        Reads :data:`ZONE_SUBREGION_CSV_PATH` and returns the subregion
        string (e.g. ``"central_north"`` or ``"south"``) associated with
        :attr:`zone`.

        Returns:
            str: Subregion name for :attr:`zone`.

        Raises:
            ValueError: If :attr:`zone` is not present in the source CSV.
        """
        zone_subregion_df = pd.read_csv(ZONE_SUBREGION_CSV_PATH, skipinitialspace=True)
        zone_subregion_df.columns = zone_subregion_df.columns.str.strip()

        df_zone = zone_subregion_df[zone_subregion_df["zone"] == self.zone]
        if df_zone.empty:
            available_zones = sorted(zone_subregion_df["zone"].astype(int).unique())
            raise ValueError(
                f"Zone {self.zone} not found in zone-subregion CSV "
                f"{ZONE_SUBREGION_CSV_PATH}. Available zones: {available_zones}."
            )

        return str(df_zone["subregion"].iloc[0]).strip()

    def _process_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate the input DataFrame to hourly and merge MISO RA-hour flags.

        Floors ``time_utc`` to the hour and averages every component column
        listed in :attr:`component_list` within each hour.  The hourly frame
        is then inner-joined to the bundled MISO RA-hour reference table on
        ``time_utc``.  The RA and AAOC flag columns selected by
        :attr:`subregion` (i.e. ``ra_<subregion>`` and ``aaoc_<subregion>``)
        are coerced to clean booleans.  ``planning_year`` (MISO planning
        year encoded as int ``YYYY``, e.g. ``2223`` for Sept 2022 -> Aug
        2023) is carried through directly from the reference table.

        Args:
            df (pd.DataFrame): DataFrame containing at minimum a
                timezone-aware ``time_utc`` column and one column per entry
                of :attr:`component_list`.

        Returns:
            pd.DataFrame: Hourly DataFrame with ``time_utc``, every column
            in :attr:`component_list` (values in MW, inherited from the
            input ``df``), the ``planning_year``, ``season``,
            ``ra_<subregion>``, and ``aaoc_<subregion>`` columns from the
            reference table.
        """
        df_hourly = (
            df.groupby(df["time_utc"].dt.floor("h"))[self.component_list]
            .mean()
            .reset_index()
        )

        df_ra = pd.read_csv(RA_HOURS_CSV_PATH)
        df_ra["time_utc"] = pd.to_datetime(df_ra["time_utc"], utc=True)

        ra_column = f"ra_{self.subregion}"
        aaoc_column = f"aaoc_{self.subregion}"

        df_merge = pd.merge(df_hourly, df_ra, on="time_utc", how="inner")

        # Coerce flag columns to clean booleans so callers can supply either
        # bool or 0/1 numeric reference tables without surprises.
        df_merge[ra_column] = _coerce_to_bool_mask(df_merge[ra_column], ra_column)
        df_merge[aaoc_column] = _coerce_to_bool_mask(df_merge[aaoc_column], aaoc_column)

        df_merge["planning_year"] = df_merge["planning_year"].astype(int)

        return df_merge

    def _limit_hourly_availability_contributions(self) -> pd.DataFrame:
        """Cap the row-wise sum of component contributions at the interconnect limit.

        Operates on :attr:`df_h_mw`, which is already hourly (one row per
        clock hour), so each row's value is itself the hourly mean and no
        sub-hourly aggregation is needed.  For every hour whose row-sum
        over :attr:`component_list` exceeds :attr:`interconnect_limit`,
        the contributions are reduced as follows:

        - When :attr:`priority_order` is ``None``, the reduction is
          *pro-rata*: every component in that hour is multiplied by the
          same factor ``interconnect_limit / row_sum`` so that the new
          row-sum equals :attr:`interconnect_limit`.
        - Otherwise :attr:`priority_order` is a dict keyed by season,
          whose values list :attr:`component_list` entries from highest
          priority (kept) to lowest priority (cut first).  For each row,
          the season's order is walked in reverse: the lowest-priority
          column absorbs as much of the excess as it can without going
          negative, and any remaining excess cascades to the
          next-lowest, and so on.

        Returns:
            pd.DataFrame: A copy of :attr:`df_h_mw` with the values in
            :attr:`component_list` reduced so that the per-hour row-sum
            is at most :attr:`interconnect_limit`.  All other columns
            (``season``, ``ra_<subregion>``, ``aaoc_<subregion>``,
            ``planning_year``, ...) are passed through unchanged.
        """
        df_h_limit = self.df_h_mw.copy()
        components = self.component_list
        limit = self.interconnect_limit_mw

        row_sum = df_h_limit[components].sum(axis=1)

        if self.priority_order is None:
            # Pro-rata: scale every component column by the same
            # row-specific factor in over-limit hours; leave others alone.
            scale = (limit / row_sum.where(row_sum > limit)).fillna(1.0)
            for col in components:
                df_h_limit[col] = df_h_limit[col] * scale
            return df_h_limit

        # Priority order: per-season list ordered from highest priority
        # (kept) to lowest priority (cut first).  Walk in reverse so the
        # lowest-priority column absorbs as much excess as it can before
        # cascading to the next-lowest.
        excess = (row_sum - limit).clip(lower=0.0)
        for season, order in self.priority_order.items():
            season_mask = df_h_limit["season"] == season
            if not season_mask.any():
                continue

            remaining = excess[season_mask]
            for col in reversed(order):
                # A column can absorb at most its own non-negative value
                # without flipping sign.
                available = df_h_limit.loc[season_mask, col].clip(lower=0.0)
                reduction = np.minimum(remaining, available)
                df_h_limit.loc[season_mask, col] = (
                    df_h_limit.loc[season_mask, col] - reduction
                )
                remaining = remaining - reduction

        return df_h_limit

    def _compute_aaoc_per_planning_year(self) -> dict[tuple[int, str], float]:
        """Compute per-PY, per-component mean availability over AAOC hours.

        Operates on :attr:`df_h_limit_mw`.  Planning years with no AAOC
        hours in :attr:`df_h_limit_mw` are simply absent from the
        returned dict; subsequent tier-2 padding falls back to ``NaN``
        for those planning years rather than raising.

        Returns:
            dict[tuple[int, str], float]: Mapping from
            ``(planning_year, component)`` to the mean of ``component``
            over AAOC hours in that planning year.
        """
        aaoc_column = f"aaoc_{self.subregion}"
        df_aaoc = self.df_h_limit_mw[self.df_h_limit_mw[aaoc_column]]
        if df_aaoc.empty:
            return {}

        per_py = df_aaoc.groupby("planning_year")[self.component_list].mean()
        return {
            (int(planning_year), component): float(per_py.loc[planning_year, component])
            for planning_year in per_py.index
            for component in self.component_list
        }

    def _compute_hour_counts(self) -> dict[tuple[int, str], int]:
        """Compute the per-(planning_year, season) hour-count diagnostic.

        Counts the classified hours present in :attr:`df_h_limit_mw` for
        each ``(planning_year, season)`` pair.  Should be called *before*
        :meth:`_drop_low_hour_planning_years` so the dict reflects the
        original coverage of the input simulation, not the post-filter
        coverage.

        Returns:
            dict[tuple[int, str], int]: Mapping from
            ``(planning_year, season)`` to the count of classified hours
            in :attr:`df_h_limit_mw`.
        """
        per_py_season = self.df_h_limit_mw.groupby(["planning_year", "season"]).size()
        return {
            (int(planning_year), str(season)): int(count)
            for (planning_year, season), count in per_py_season.items()
        }

    def _drop_low_hour_planning_years(self) -> pd.DataFrame:
        """Drop sparsely-covered planning years from :attr:`df_h_limit_mw`.

        When :attr:`remove_low_hour_planning_years` is True, any planning
        year whose total classified-hour count in :attr:`df_h_limit_mw`
        is below :data:`_LOW_HOUR_PLANNING_YEAR_THRESHOLD_HOURS` is
        dropped *whole* (all four seasons together) so partial coverage
        cannot contaminate the per-component availability statistics.
        :attr:`hours_per_planning_year_season` itself is left untouched
        as a diagnostic of the *original* coverage; callers can compare
        it against the returned frame to see what was dropped.

        When :attr:`remove_low_hour_planning_years` is False, the input
        :attr:`df_h_limit_mw` is returned unchanged.

        Returns:
            pd.DataFrame: The filtered (or unchanged)
            :attr:`df_h_limit_mw`.
        """
        if not self.remove_low_hour_planning_years:
            return self.df_h_limit_mw

        threshold_hours = _LOW_HOUR_PLANNING_YEAR_THRESHOLD_HOURS
        df = self.df_h_limit_mw
        py_sizes = df.groupby("planning_year")["planning_year"].transform("size")
        keep_mask = py_sizes >= threshold_hours
        if keep_mask.all():
            return df

        if self.verbose:
            py_totals = df.groupby("planning_year").size()
            dropped = sorted(
                int(py) for py, count in py_totals.items() if count < threshold_hours
            )
            print(
                f"[MisoCapacity] dropping {len(dropped)} planning year(s) "
                f"with < {threshold_hours} hours: {dropped}\n"
                f"  - Startdate moved from {df.iloc[0]['time_utc']} to {df.loc[keep_mask].iloc[0]['time_utc']}\n"
                f"  - Enddate moved from {df.iloc[-1]['time_utc']} to {df.loc[keep_mask].iloc[-1]['time_utc']}\n"
            )
        return df.loc[keep_mask].reset_index(drop=True)

    def _compute_tier_availabilities(
        self,
    ) -> tuple[
        dict[tuple[str, str], float],
        dict[tuple[str, str], float],
        dict[tuple[str, str], float],
        dict[tuple[str, str], float],
    ]:
        """Compute per-(season, component) tier 1/2, all-hours, and ISAC means.

        Operates on :attr:`df_h_limit_mw` and assumes
        :attr:`aaoc_per_planning_year_mw` has already been populated by
        :meth:`_compute_aaoc_per_planning_year`.

        For each ``(season, component)`` pair, four values are produced:

        - ``tier_1_availability``: mean of ``component`` over hours
          that are NOT seasonal RA hours, pooled across every planning
          year present in :attr:`df_h_limit_mw`.
        - ``tier_2_availability``: mean of ``component`` over seasonal
          RA hours, pooled across planning years.  For any
          ``(planning_year, season)`` combination with fewer than 65
          RA hours, that planning year's contribution is right-padded
          up to 65 entries using
          ``self.aaoc_per_planning_year_mw[(planning_year, component)]``
          before being concatenated.  Combinations with zero classified
          hours in :attr:`df_h_limit_mw` are skipped entirely (no
          synthetic AAOC padding).
        - ``all_hours_availability``: mean of ``component`` over every
          classified hour in the season.
        - ``isac_mw``: ``0.2 * tier_1_mw + 0.8 * tier_2_mw``.

        When a ``(planning_year, season)`` combination has fewer than
        65 RA hours and the same planning year has no AAOC entry in
        :attr:`aaoc_per_planning_year_mw`, padding falls back to
        ``NaN``; the resulting tier-2 mean (and therefore ISAC) for any
        ``(season, component)`` whose data depends on that planning
        year propagates as ``NaN``, making partial-PY coverage visible
        to the caller via the output dicts.

        Returns:
            tuple[dict, dict, dict, dict]: A 4-tuple
            ``(tier_1_availability_mw, tier_2_availability_mw,
            all_hours_availability_mw, isac_mw)`` of dicts keyed by
            ``(season, component)``.
        """
        ra_column = f"ra_{self.subregion}"
        df = self.df_h_limit_mw
        seasons = df["season"].unique()
        planning_years = sorted(int(py) for py in df["planning_year"].unique())

        tier_1_availability: dict[tuple[str, str], float] = {}
        tier_2_availability: dict[tuple[str, str], float] = {}
        all_hours_availability: dict[tuple[str, str], float] = {}
        isac: dict[tuple[str, str], float] = {}

        for season in seasons:
            df_season = df[df["season"] == season]
            for component in self.component_list:
                tier_1_values: list[np.ndarray] = []
                tier_2_values: list[np.ndarray] = []

                for planning_year in planning_years:
                    df_py_season = df_season[
                        df_season["planning_year"] == planning_year
                    ]
                    if df_py_season.empty:
                        # Skip (planning_year, season) pairs with no
                        # classified hours so partial PYs don't inject
                        # synthetic AAOC-padded entries into tier 2.
                        continue

                    tier_1_values.append(
                        df_py_season.loc[~df_py_season[ra_column], component].to_numpy()
                    )
                    tier_2_py = df_py_season.loc[
                        df_py_season[ra_column], component
                    ].to_numpy()
                    if len(tier_2_py) < _TIER_2_PAD_TARGET:
                        pad_value = self.aaoc_per_planning_year_mw.get(
                            (planning_year, component), float("nan")
                        )
                        tier_2_py = np.pad(
                            tier_2_py,
                            (0, _TIER_2_PAD_TARGET - len(tier_2_py)),
                            mode="constant",
                            constant_values=pad_value,
                        )
                    tier_2_values.append(tier_2_py)

                tier_1_arr = (
                    np.concatenate(tier_1_values) if tier_1_values else np.array([])
                )
                tier_2_arr = (
                    np.concatenate(tier_2_values) if tier_2_values else np.array([])
                )

                tier_1_mean = (
                    float(np.mean(tier_1_arr)) if tier_1_arr.size else float("nan")
                )
                tier_2_mean = (
                    float(np.mean(tier_2_arr)) if tier_2_arr.size else float("nan")
                )
                all_mean = float(df_season[component].mean())

                key = (str(season), component)
                tier_1_availability[key] = tier_1_mean
                tier_2_availability[key] = tier_2_mean
                all_hours_availability[key] = all_mean
                isac[key] = 0.2 * tier_1_mean + 0.8 * tier_2_mean

        return tier_1_availability, tier_2_availability, all_hours_availability, isac

    def _load_class_ucap_mw(self) -> dict[tuple[str, str], float]:
        """Load class-level UCAP per ``(season, component)`` in MW.

        Reads :data:`RESOURCE_CLASS_UCAP_CSV_PATH` and returns a dict
        keyed by ``(season, component)`` with values in MW.  Each
        component is assigned the UCAP MW of its associated resource
        class from :attr:`class_list`, evaluated for every season in
        :data:`_SEASONS`.

        Returns:
            dict[tuple[str, str], float]: Mapping from
            ``(season, component)`` to class-level UCAP in MW.

        Raises:
            ValueError: If a resource class in :attr:`class_list` is not
                present in the source CSV.
        """
        ucap_df = pd.read_csv(RESOURCE_CLASS_UCAP_CSV_PATH)
        ucap_df["resource_class"] = ucap_df["resource_class"].astype(str).str.strip()
        ucap_df = ucap_df.set_index("resource_class")

        class_ucap_mw: dict[tuple[str, str], float] = {}
        for component, resource_class in zip(
            self.component_list, self.class_list, strict=True
        ):
            if resource_class not in ucap_df.index:
                raise ValueError(
                    f"Resource class '{resource_class}' not found in UCAP CSV "
                    f"{RESOURCE_CLASS_UCAP_CSV_PATH}."
                )
            for season in _SEASONS:
                class_ucap_mw[(season, component)] = float(
                    ucap_df.loc[resource_class, season]
                )
        return class_ucap_mw

    def _load_class_isac_mw(self) -> dict[tuple[str, str], float]:
        """Load class-level ISAC per ``(season, component)`` in MW.

        Reads :data:`RESOURCE_CLASS_ISAC_CSV_PATH` and returns a dict
        keyed by ``(season, component)`` with values in MW.  Each
        component is assigned the ISAC MW of its associated resource
        class from :attr:`class_list`, evaluated for every season in
        :data:`_SEASONS`.

        Returns:
            dict[tuple[str, str], float]: Mapping from
            ``(season, component)`` to class-level ISAC in MW.

        Raises:
            ValueError: If a resource class in :attr:`class_list` is not
                present in the source CSV.
        """
        isac_df = pd.read_csv(RESOURCE_CLASS_ISAC_CSV_PATH)
        isac_df["resource_class"] = isac_df["resource_class"].astype(str).str.strip()
        isac_df = isac_df.set_index("resource_class")

        class_isac_mw: dict[tuple[str, str], float] = {}
        for component, resource_class in zip(
            self.component_list, self.class_list, strict=True
        ):
            if resource_class not in isac_df.index:
                raise ValueError(
                    f"Resource class '{resource_class}' not found in ISAC CSV "
                    f"{RESOURCE_CLASS_ISAC_CSV_PATH}."
                )
            for season in _SEASONS:
                class_isac_mw[(season, component)] = float(
                    isac_df.loc[resource_class, season]
                )
        return class_isac_mw

    def _compute_class_ucap_isac_conversion(self) -> dict[tuple[str, str], float]:
        """Compute the class-level UCAP / ISAC conversion factor.

        Assumes :attr:`class_ucap_mw` and :attr:`class_isac_mw` have already
        been populated over the same ``(season, component)`` keys.
        Returns the per-key ratio ``class_ucap_mw / class_isac_mw`` as a dict
        keyed by ``(season, component)``.  This factor can later be
        applied to a component-level ISAC to convert it to a
        component-level UCAP.

        ``(season, component)`` keys whose ISAC value is zero map to
        ``NaN`` rather than raising, so callers can distinguish
        undefined conversions from valid zero-ratio ones.

        Returns:
            dict[tuple[str, str], float]: Mapping from
            ``(season, component)`` to ``class_ucap / class_isac``
            (or ``NaN`` when ``class_isac`` is zero).
        """
        class_ucap_isac_conversion: dict[tuple[str, str], float] = {}
        for key, isac_value in self.class_isac_mw.items():
            ucap_value = self.class_ucap_mw[key]
            if isac_value == 0.0:
                class_ucap_isac_conversion[key] = float("nan")
            else:
                class_ucap_isac_conversion[key] = ucap_value / isac_value
        return class_ucap_isac_conversion

    def _compute_sac_mw(self) -> dict[tuple[str, str], float]:
        """Compute the component-level Seasonal Accredited Capacity (SAC) in MW.

        Assumes :attr:`isac_mw` and :attr:`class_ucap_isac_conversion`
        have already been populated.  For every ``(season, component)``
        key present in :attr:`isac_mw`, the SAC is computed as
        ``isac_mw * class_ucap_isac_conversion``.  Because
        :attr:`isac_mw` is already in MW and :attr:`class_ucap_isac_conversion`
        is a dimensionless MW/MW ratio, the result is in MW with no
        additional unit conversion.

        The returned dict shares its keys with :attr:`isac_mw` (i.e. only
        seasons that are present in :attr:`df_h_limit_mw`).  When a key is
        missing from :attr:`class_ucap_isac_conversion` (which should
        not happen because the conversion dict spans all four seasons),
        the SAC value falls back to ``NaN`` so partial coverage stays
        visible to the caller.

        Returns:
            dict[tuple[str, str], float]: Mapping from
            ``(season, component)`` to component-level SAC in MW.
        """
        sac_mw: dict[tuple[str, str], float] = {}
        for key, isac_value in self.isac_mw.items():
            conversion = self.class_ucap_isac_conversion.get(key, float("nan"))
            sac_mw[key] = isac_value * conversion
        return sac_mw

    def _get_available_pra_years(self) -> list[int]:
        """Get available PRA price years from the bundled PRA CSV.

        Reads :data:`PRA_PRICES_CSV_PATH` and returns sorted unique values
        from the ``year`` column.

        Returns:
            list[int]: Sorted list of available PRA years.
        """
        pra_prices_df = pd.read_csv(PRA_PRICES_CSV_PATH)
        return sorted(pra_prices_df["year"].dropna().astype(int).unique().tolist())

    def _get_pra_prices(self) -> dict[str, float]:
        """Get the PRA clearing prices per season for :attr:`zone`.

        Reads :data:`PRA_PRICES_CSV_PATH`, filters to :attr:`zone` and
        :attr:`pra_years`, and returns a dict keyed by season with values
        in $/MW-day.  When :attr:`pra_years` contains more than one
        year, the prices are averaged across years (per season).

        Returns:
            dict[str, float]: Mapping from season name (``"summer"``,
            ``"fall"``, ``"winter"``, ``"spring"``) to PRA clearing price
            in $/MW-day.

        Raises:
            ValueError: If :attr:`zone` is missing from the source CSV
                or if any year in :attr:`pra_years` has no row for
                :attr:`zone`.
        """
        pra_prices_df = pd.read_csv(PRA_PRICES_CSV_PATH)

        df_zone = pra_prices_df[pra_prices_df["zone"] == self.zone]
        if df_zone.empty:
            raise ValueError(
                f"Zone {self.zone} not found in PRA prices CSV {PRA_PRICES_CSV_PATH}."
            )

        df_filtered = df_zone[df_zone["year"].isin(self.pra_years)]
        missing_years = sorted(set(self.pra_years) - set(df_filtered["year"].tolist()))
        if missing_years:
            raise ValueError(
                f"PRA years {missing_years} not found for zone {self.zone} in "
                f"PRA prices CSV {PRA_PRICES_CSV_PATH}."
            )

        season_means = df_filtered[list(_SEASONS)].mean(axis=0)
        return {season: float(season_means[season]) for season in _SEASONS}

    def _get_days_per_season(self) -> dict[str, float]:
        """Get average seasonal day counts across :attr:`pra_years`.

        Reads the appropriate days-per-season CSV for each year in
        :attr:`pra_years` (choosing the leap-year table when the year is a
        leap year) and returns the average day count per season.  When
        :attr:`pra_years` contains a single year the result is the exact
        integer count for that year cast to float; with multiple years the
        result is a weighted average and may be fractional.

        Returns:
            dict[str, float]: Mapping from season name (``"summer"``,
            ``"fall"``, ``"winter"``, ``"spring"``) to average number of
            days in that season across :attr:`pra_years`.

        Raises:
            KeyError: If any expected season key is missing from the source
                table.
        """
        all_days: list[dict[str, int]] = []
        for year in self.pra_years:
            path = (
                DAYS_PER_SEASON_LEAP_YEAR_CSV_PATH
                if calendar.isleap(year)
                else DAYS_PER_SEASON_CSV_PATH
            )
            days_by_season = pd.read_csv(path).set_index("season")["days"]
            try:
                all_days.append(
                    {season: int(days_by_season.loc[season]) for season in _SEASONS}
                )
            except KeyError as err:
                raise KeyError(
                    f"Missing season day count in source table. "
                    f"Expected seasons: {list(_SEASONS)}; "
                    f"available seasons: {sorted(days_by_season.index)}."
                ) from err
        return {
            season: sum(d[season] for d in all_days) / len(all_days)
            for season in _SEASONS
        }

    def _compute_revenue_per_season(self) -> dict[tuple[str, str], float]:
        """Compute per-(season, component) capacity revenue.

        Multiplies the component ZRC (Zonal Resource Credits, MW) by the
        season's PRA clearing price ($/MW-day) and the number of days in
        that season.  Only ``(season, component)`` pairs present in
        :attr:`zrc_mw` contribute; seasons absent from :attr:`df_h_limit_mw` (e.g.
        because they were filtered by :meth:`_drop_low_hour_planning_years`) will
        not appear in the returned dict.

        Assumes :attr:`zrc_mw`, :attr:`pra_prices`, and :attr:`days_per_season`
        have already been populated.

        Returns:
            dict[tuple[str, str], float]: Mapping from
            ``(season, component)`` to revenue in dollars.
        """
        return {
            (season, component): zrc_value
            * self.pra_prices[season]
            * self.days_per_season[season]
            for (season, component), zrc_value in self.zrc_mw.items()
        }

    def _compute_annual_revenue(self) -> dict[str, float]:
        """Compute total annual revenue per component, summed across seasons.

        Sums :attr:`revenue_per_season` over all four seasons for each
        component.  Seasons absent from :attr:`revenue_per_season` contribute
        zero to the sum, so a component with partial-year data will show a
        proportionally lower annual total.

        Assumes :attr:`revenue_per_season` has already been populated.

        Returns:
            dict[str, float]: Mapping from component name to total annual
            revenue in dollars.
        """
        return {
            component: sum(
                self.revenue_per_season.get((season, component), 0.0)
                for season in _SEASONS
            )
            for component in self.component_list
        }

    def get_component_table(self, component: str) -> pd.DataFrame:
        """Build a per-season results table for a single component.

        The returned DataFrame has one column per MISO season (summer, fall,
        winter, spring) and one row per metric, in the same order as the
        :meth:`__init__` computation flow.  Seasons absent from
        :attr:`df_h_limit_mw` (e.g. filtered by
        :meth:`_drop_low_hour_planning_years`) show ``NaN``.

        Args:
            component (str): A component name from :attr:`component_list`.

        Returns:
            pd.DataFrame: Shape ``(11, 4)`` with seasons as columns and
            metrics as the index.

        Raises:
            ValueError: If ``component`` is not in :attr:`component_list`.
        """
        if component not in self.component_list:
            raise ValueError(
                f"Component '{component}' not in component_list: {self.component_list}."
            )

        hours_per_season = (
            self.df_h_limit_mw.groupby("season").size().to_dict()
            if not self.df_h_limit_mw.empty
            else {}
        )

        rows: dict[str, dict[str, float]] = {
            "# Hours": {},
            "Tier 1 (MW)": {},
            "Tier 2 (MW)": {},
            "ISAC (MW)": {},
            "Class UCAP (MW)": {},
            "Class ISAC (MW)": {},
            "SAC (MW)": {},
            "ZRC (MW)": {},
            "PRA Price ($/MW-day)": {},
            "Days": {},
            "Revenue ($)": {},
        }
        for season in _SEASONS:
            key = (season, component)
            rows["# Hours"][season] = float(hours_per_season.get(season, float("nan")))
            rows["Tier 1 (MW)"][season] = self.tier_1_availability_mw.get(
                key, float("nan")
            )
            rows["Tier 2 (MW)"][season] = self.tier_2_availability_mw.get(
                key, float("nan")
            )
            rows["ISAC (MW)"][season] = self.isac_mw.get(key, float("nan"))
            rows["Class UCAP (MW)"][season] = self.class_ucap_mw.get(key, float("nan"))
            rows["Class ISAC (MW)"][season] = self.class_isac_mw.get(key, float("nan"))
            rows["SAC (MW)"][season] = self.sac_mw.get(key, float("nan"))
            rows["ZRC (MW)"][season] = self.zrc_mw.get(key, float("nan"))
            rows["PRA Price ($/MW-day)"][season] = self.pra_prices.get(
                season, float("nan")
            )
            rows["Days"][season] = self.days_per_season.get(season, float("nan"))
            rows["Revenue ($)"][season] = self.revenue_per_season.get(key, float("nan"))

        return pd.DataFrame(rows, index=list(_SEASONS)).T

    def print_component_table(self, component: str) -> None:
        """Print a formatted per-season results table for a single component.

        Calls :meth:`get_component_table` and prints the result to stdout
        with a component header and comma-separated numeric formatting.

        Args:
            component (str): A component name from :attr:`component_list`.
        """
        table = self.get_component_table(component)
        print(f"\n=== {component} ===")
        print(table.to_string(float_format="{:,.2f}".format))

    def get_totals_table(self) -> pd.DataFrame:
        """Build a totals table summing capacity and revenue across all components.

        Produces the same row/column structure as :meth:`get_component_table`
        but aggregates across :attr:`component_list`.  Capacity and revenue
        rows (Tier 1–ZRC and Revenue) are summed; the ``# Hours``,
        ``PRA Price``, and ``Days`` rows are taken from the first component
        (they are identical for every component in a given season).

        Returns:
            pd.DataFrame: Shape ``(11, 4)`` with seasons as columns.
        """
        summed_rows = (
            "Tier 1 (MW)",
            "Tier 2 (MW)",
            "ISAC (MW)",
            "Class UCAP (MW)",
            "Class ISAC (MW)",
            "SAC (MW)",
            "ZRC (MW)",
            "Revenue ($)",
        )
        total: pd.DataFrame | None = None
        for component in self.component_list:
            comp_df = self.get_component_table(component)
            if total is None:
                total = comp_df.copy()
            else:
                total.loc[list(summed_rows)] += comp_df.loc[list(summed_rows)]

        return total if total is not None else pd.DataFrame()

    def print_totals_table(self) -> None:
        """Print a formatted totals table aggregated across all components.

        Calls :meth:`get_totals_table` and prints the result to stdout.
        """
        table = self.get_totals_table()
        print("\n=== TOTALS (all components) ===")
        print(table.to_string(float_format="{:,.2f}".format))

    def print_all_component_tables(self) -> None:
        """Print per-component tables followed by the fleet totals table.

        Calls :meth:`print_component_table` for each entry in
        :attr:`component_list`, then calls :meth:`print_totals_table`.
        """
        for component in self.component_list:
            self.print_component_table(component)
        self.print_totals_table()

    def get_total_revenue(self) -> float:
        """Return the total annual revenue summed across all components.

        Returns:
            float: Total annual capacity revenue in dollars.
        """
        return sum(self.annual_revenue.values())

    def get_component_revenue(self, component: str) -> float:
        """Return the total annual revenue for a single component.

        Args:
            component (str): A component name from :attr:`component_list`.

        Returns:
            float: Annual capacity revenue in dollars for ``component``.

        Raises:
            ValueError: If ``component`` is not in :attr:`component_list`.
        """
        if component not in self.component_list:
            raise ValueError(
                f"Component '{component}' not in component_list: {self.component_list}."
            )
        return self.annual_revenue[component]

    def _plot_stacked_bar_by_season(
        self,
        values: dict[tuple[str, str], float],
        ylabel: str,
        title: str,
        bar_label_fmt: str,
        ax: "plt.Axes | None" = None,
        save_path: Path | None = None,
    ) -> "plt.Figure":
        """Render a per-season stacked bar plot keyed by ``(season, component)``.

        Shared backend for :meth:`plot_sac_stacked_bar` and
        :meth:`plot_revenue_stacked_bar`.  Missing ``(season, component)``
        entries are treated as zero.

        Args:
            values (dict[tuple[str, str], float]): Mapping from
                ``(season, component)`` to the per-season value to plot.
            ylabel (str): Y-axis label.
            title (str): Axes title.
            bar_label_fmt (str): Format string applied to each season's
                stacked total, e.g. ``"{:,.1f}"`` or ``"${:,.0f}"``.
            ax (plt.Axes | None): Existing axes to draw into.  When
                ``None`` a new figure/axes pair is created.
            save_path (Path | None): If provided, save the figure here.

        Returns:
            plt.Figure: The matplotlib figure containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.5))
        else:
            fig = ax.figure

        seasons = list(_SEASONS)
        x = np.arange(len(seasons))
        bottoms = np.zeros(len(seasons), dtype=float)
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for idx, component in enumerate(self.component_list):
            heights = np.array(
                [
                    float(values.get((season, component), 0.0) or 0.0)
                    for season in seasons
                ]
            )
            ax.bar(
                x,
                heights,
                bottom=bottoms,
                label=component,
                color=color_cycle[idx % len(color_cycle)],
            )
            bottoms = bottoms + heights

        for xi, total in zip(x, bottoms, strict=True):
            ax.text(
                xi,
                total,
                bar_label_fmt.format(total),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in seasons])
        ax.set_xlabel("Season")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.margins(y=0.12)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)
        return fig

    def plot_sac_stacked_bar(
        self,
        ax: "plt.Axes | None" = None,
        save_path: Path | None = None,
    ) -> "plt.Figure":
        """Plot per-season SAC as a stacked bar chart, colored by component.

        For each MISO season (summer, fall, winter, spring) draws a single
        bar whose height is the total Seasonal Accredited Capacity (MW)
        across all components in :attr:`component_list`, stacked and
        colored by component.  The total height is annotated above each
        bar.

        Args:
            ax (plt.Axes | None): Existing axes to draw into.  When
                ``None`` a new figure/axes pair is created.
            save_path (Path | None): If provided, save the figure here.

        Returns:
            plt.Figure: The matplotlib figure containing the plot.
        """
        return self._plot_stacked_bar_by_season(
            values=self.sac_mw,
            ylabel="SAC (MW)",
            title="Seasonal Accredited Capacity by component",
            bar_label_fmt="{:,.1f}",
            ax=ax,
            save_path=save_path,
        )

    def plot_revenue_stacked_bar(
        self,
        ax: "plt.Axes | None" = None,
        save_path: Path | None = None,
    ) -> "plt.Figure":
        """Plot per-season capacity revenue as a stacked bar chart.

        For each MISO season (summer, fall, winter, spring) draws a single
        bar whose height is the total capacity revenue (USD) across all
        components in :attr:`component_list`, stacked and colored by
        component.  The total height is annotated above each bar.

        Args:
            ax (plt.Axes | None): Existing axes to draw into.  When
                ``None`` a new figure/axes pair is created.
            save_path (Path | None): If provided, save the figure here.

        Returns:
            plt.Figure: The matplotlib figure containing the plot.
        """
        return self._plot_stacked_bar_by_season(
            values=self.revenue_per_season,
            ylabel="Revenue ($)",
            title="Capacity revenue by component",
            bar_label_fmt="${:,.0f}",
            ax=ax,
            save_path=save_path,
        )

    def plot_isac_availability_bar(
        self,
        component: str,
        ax: "plt.Axes | None" = None,
        save_path: Path | None = None,
    ) -> "plt.Figure":
        """Plot a grouped bar chart of Tier 1, Tier 2, and ISAC availability by season.

        For each MISO season in which ``component`` has computed ISAC values,
        draws three side-by-side bars showing the Tier 1 (non-RA hours), Tier 2
        (RA hours), and ISAC availability in MW.  A horizontal dashed line marks
        the mean AAOC across all planning years.

        Args:
            component (str): Component name; must appear in
                :attr:`component_list`.
            ax (plt.Axes | None): Existing axes to draw into.  When ``None``
                a new figure/axes pair is created.  Defaults to None.
            save_path (Path | None): If provided, save the figure here.
                Defaults to None.

        Returns:
            plt.Figure: The matplotlib figure containing the bar chart.

        Raises:
            ValueError: If ``component`` is not in :attr:`component_list`.
        """
        if component not in self.component_list:
            raise ValueError(
                f"'{component}' not found in component_list: {self.component_list}"
            )

        seasons = [s for s in _SEASONS if (s, component) in self.isac_mw]

        tier1 = [self.tier_1_availability_mw[(s, component)] for s in seasons]
        tier2 = [self.tier_2_availability_mw[(s, component)] for s in seasons]
        isac = [self.isac_mw[(s, component)] for s in seasons]

        aaoc_values = [
            v
            for (_, comp), v in self.aaoc_per_planning_year_mw.items()
            if comp == component
        ]
        aaoc_value = float(np.mean(aaoc_values)) if aaoc_values else None

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))
        else:
            fig = ax.figure

        x = np.arange(len(seasons))
        width = 0.25

        ax.bar(
            x - width, tier1, width, label="Tier 1 (non-RA hours)", color="steelblue"
        )
        ax.bar(x, tier2, width, label="Tier 2 (RA hours)", color="darkorange")
        ax.bar(x + width, isac, width, label="ISAC (0.2·T1 + 0.8·T2)", color="seagreen")
        if aaoc_value is not None:
            ax.axhline(
                aaoc_value,
                color="purple",
                ls="--",
                lw=1.5,
                label=f"AAOC ({aaoc_value:.1f} MW)",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in seasons])
        ax.set_ylabel("Availability (MW)")
        ax.set_title(f"{component} — seasonal Tier 1 / Tier 2 / ISAC availability")
        ax.legend()
        ax.grid(axis="y")
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)
        return fig

    def plot_isac_hourly_scatter(
        self,
        component: str,
        axes: "list[plt.Axes] | None" = None,
        save_path: Path | None = None,
    ) -> "plt.Figure":
        """Plot hourly output scatter by season with RA hours and ISAC metrics overlaid.

        For each MISO season in which ``component`` has computed ISAC values,
        draws a scatter plot of hourly ``component`` output (MW) colored by
        whether each hour is an RA hour (Tier 2) or non-RA hour (Tier 1).
        Horizontal reference lines are added for Tier 1 availability, Tier 2
        availability, ISAC, and the mean AAOC across all planning years.

        Args:
            component (str): Component name; must appear in
                :attr:`component_list`.
            axes (list[plt.Axes] | None): Pre-created list of axes, one per
                available season (in the order summer → fall → winter →
                spring).  When ``None`` a new figure/axes row is created.
            save_path (Path | None): If provided, save the figure here.
                Defaults to None.

        Returns:
            plt.Figure: The matplotlib figure containing the scatter plots.

        Raises:
            ValueError: If ``component`` is not in :attr:`component_list`.
        """
        if component not in self.component_list:
            raise ValueError(
                f"'{component}' not found in component_list: {self.component_list}"
            )

        seasons = [s for s in _SEASONS if (s, component) in self.isac_mw]

        season_color = {
            "summer": "tab:orange",
            "fall": "tab:brown",
            "winter": "tab:blue",
            "spring": "tab:green",
        }

        if axes is None:
            fig, axes = plt.subplots(
                1,
                len(seasons),
                figsize=(5 * len(seasons), 5),
                sharey=True,
            )
            if len(seasons) == 1:
                axes = [axes]
        else:
            fig = axes[0].figure

        ra_col = f"ra_{self.subregion}"
        df_h = self.df_h_limit_mw

        # Average AAOC across all planning years for this component
        aaoc_values = [
            v
            for (_, comp), v in self.aaoc_per_planning_year_mw.items()
            if comp == component
        ]
        aaoc_value = float(np.mean(aaoc_values)) if aaoc_values else None

        for ax, season in zip(axes, seasons, strict=False):
            df_s = df_h[df_h["season"] == season].reset_index(drop=True)
            ra_mask = df_s[ra_col]

            tier1_hours = int((~ra_mask).sum())
            tier2_hours = int(ra_mask.sum())

            ax.scatter(
                df_s.index[~ra_mask],
                df_s.loc[~ra_mask, component],
                s=2,
                alpha=0.3,
                color="steelblue",
                label=f"Non-RA hours (Tier 1: {tier1_hours}h)",
            )
            ax.scatter(
                df_s.index[ra_mask],
                df_s.loc[ra_mask, component],
                s=6,
                alpha=0.7,
                color="tomato",
                label=f"RA hours (Tier 2: {tier2_hours}h)",
            )

            key = (season, component)
            ax.axhline(
                self.tier_1_availability_mw[key],
                color="steelblue",
                ls="--",
                lw=1.5,
                label=f"Tier 1 ({self.tier_1_availability_mw[key]:.1f} MW)",
            )
            ax.axhline(
                self.tier_2_availability_mw[key],
                color="darkorange",
                ls="--",
                lw=1.5,
                label=f"Tier 2 ({self.tier_2_availability_mw[key]:.1f} MW)",
            )
            ax.axhline(
                self.isac_mw[key],
                color=season_color[season],
                ls="-",
                lw=2,
                label=f"ISAC  ({self.isac_mw[key]:.1f} MW)",
            )
            if aaoc_value is not None:
                ax.axhline(
                    aaoc_value,
                    color="purple",
                    ls="--",
                    lw=1.5,
                    label=f"AAOC  ({aaoc_value:.1f} MW)",
                )

            ax.set_title(season.capitalize())
            ax.set_xlabel("Hour index (within season)")
            if ax is axes[0]:
                ax.set_ylabel(f"{component} (MW)")
            ax.legend(markerscale=4, fontsize=8)

        fig.suptitle(
            f"Hourly output — RA hours highlighted with ISAC metrics\n{component}"
        )
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)
        return fig

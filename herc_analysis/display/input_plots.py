"""Helpers for exploring time-series input signals.

This module provides DataFrame-first plotting and summary helpers for the
input signals that typically drive a Hercules simulation — market prices
(real-time / day-ahead LMPs), wind speed, solar irradiance, and similar
time series.

All functions assume the input DataFrame contains a ``time_utc`` column with
timezone-aware UTC timestamps (as produced by ``pd.to_datetime(..., utc=True)``
or by ``OutputAnalysis``).  Functions never mutate the input DataFrame.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Default transparent red / blue used for negative / positive price wedges.
DEFAULT_DONUT_COLORS: tuple[str, str] = ("#FF000050", "#0000FF50")

# Heuristic: column names matching any of these substrings get default price
# histogram bins applied (in $/MWh).
_PRICE_NAME_HINTS = ("lmp", "price")
_DEFAULT_PRICE_BINS = np.arange(-100, 101, 1)


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------


def _validate_time_utc(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if *df* lacks a usable ``time_utc`` column."""
    if "time_utc" not in df.columns:
        raise ValueError("DataFrame must contain a 'time_utc' column.")
    if not pd.api.types.is_datetime64_any_dtype(df["time_utc"]):
        raise TypeError(
            "Column 'time_utc' must be a datetime dtype (e.g. produced by "
            "pd.to_datetime(..., utc=True))."
        )


def _looks_like_price(value_col: str) -> bool:
    """Return ``True`` if *value_col* name suggests a market price."""
    lc = value_col.lower()
    return any(hint in lc for hint in _PRICE_NAME_HINTS)


def _with_year(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a ``year`` column derived from ``time_utc``."""
    out = df.copy()
    out["year"] = out["time_utc"].dt.year
    return out


# ----------------------------------------------------------------------
# Donut plots
# ----------------------------------------------------------------------


def _draw_single_donut(ax, neg_count, pos_count, colors, title=None):
    """Draw one negative/positive donut wedge on *ax*."""
    sizes = [neg_count, pos_count]
    labels = ["Negative", "Positive"]
    if sum(sizes) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        if title is not None:
            ax.set_title(title)
        return
    ax.pie(
        sizes,
        labels=labels,
        startangle=90,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%",
        pctdistance=0.8,
        textprops={"fontsize": 12},
        wedgeprops={"width": 0.4},
    )
    ax.add_artist(plt.Circle((0, 0), 0.50, fc="white"))
    if title is not None:
        ax.set_title(title)


def plot_price_donut(
    df: pd.DataFrame,
    value_col: str = "lmp_rt",
    by_year: bool = False,
    colors: tuple[str, str] | None = None,
    title: str | None = None,
):
    """Plot a donut showing the share of negative vs. positive prices.

    Args:
        df (pd.DataFrame): Input DataFrame with ``time_utc`` and *value_col*.
        value_col (str, optional): Price column. Defaults to ``"lmp_rt"``.
        by_year (bool, optional): If True, draws one donut per calendar year
            in a single row of subplots. Defaults to False.
        colors (tuple[str, str], optional): ``(negative, positive)`` wedge
            colors. Defaults to transparent red/blue.
        title (str, optional): Figure suptitle (or axis title when single).

    Returns:
        tuple: ``(fig, ax_or_axes)``. ``axes`` is an ``ndarray`` when
        ``by_year=True``.
    """
    _validate_time_utc(df)
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not in DataFrame.")
    colors = colors if colors is not None else DEFAULT_DONUT_COLORS

    if not by_year:
        fig, ax = plt.subplots(figsize=(6, 6))
        series = df[value_col].dropna()
        _draw_single_donut(
            ax,
            int((series < 0).sum()),
            int((series >= 0).sum()),
            colors,
            title=title or f"{value_col}: negative vs positive",
        )
        return fig, ax

    dfy = _with_year(df)
    years = sorted(dfy["year"].dropna().unique().astype(int))
    n = len(years)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    axes = axes[0]
    for ax, year in zip(axes, years, strict=True):
        series = dfy.loc[dfy["year"] == year, value_col].dropna()
        _draw_single_donut(
            ax,
            int((series < 0).sum()),
            int((series >= 0).sum()),
            colors,
            title=f"{year}",
        )
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


# ----------------------------------------------------------------------
# Histogram
# ----------------------------------------------------------------------


def plot_histogram(
    df: pd.DataFrame,
    value_col: str,
    bins=None,
    by_year: bool = False,
    ax=None,
    title: str | None = None,
    xlabel: str | None = None,
):
    """Plot a histogram of *value_col*.

    Args:
        df (pd.DataFrame): Input DataFrame with ``time_utc`` and *value_col*.
        value_col (str): Column to histogram.
        bins (array-like or int, optional): Passed to seaborn. When omitted
            and *value_col* looks like a price (name contains "lmp" or
            "price"), defaults to 1 $/MWh bins from -100 to 100; otherwise
            seaborn auto-bins.
        by_year (bool, optional): If True, overlays one histogram per year
            using ``hue="year"``. Defaults to False.
        ax (matplotlib.axes.Axes, optional): Axes to draw on. A new figure
            is created if not supplied.
        title (str, optional): Axes title.
        xlabel (str, optional): X-axis label. Defaults to *value_col*.

    Returns:
        tuple: ``(fig, ax)``.
    """
    _validate_time_utc(df)
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not in DataFrame.")

    if bins is None and _looks_like_price(value_col):
        bins = _DEFAULT_PRICE_BINS

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    hist_kwargs = {"ax": ax}
    if bins is not None:
        hist_kwargs["bins"] = bins

    if by_year:
        dfy = _with_year(df).dropna(subset=[value_col])
        sns.histplot(
            data=dfy,
            x=value_col,
            hue="year",
            element="step",
            stat="count",
            **hist_kwargs,
        )
    else:
        sns.histplot(df[value_col].dropna(), **hist_kwargs)

    ax.set_xlabel(xlabel if xlabel is not None else value_col)
    ax.set_title(title or f"Histogram of {value_col}")
    return fig, ax


# ----------------------------------------------------------------------
# Boxplot by year
# ----------------------------------------------------------------------


def plot_boxplot_by_year(
    df: pd.DataFrame,
    value_col: str,
    showfliers: bool = False,
    ax=None,
    title: str | None = None,
):
    """Plot a per-year boxplot of *value_col*.

    Args:
        df (pd.DataFrame): Input DataFrame with ``time_utc`` and *value_col*.
        value_col (str): Column to summarize.
        showfliers (bool, optional): Show outlier markers. Defaults to False.
        ax (matplotlib.axes.Axes, optional): Axes to draw on.
        title (str, optional): Axes title.

    Returns:
        tuple: ``(fig, ax)``.
    """
    _validate_time_utc(df)
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not in DataFrame.")

    dfy = _with_year(df).dropna(subset=[value_col])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    sns.boxplot(data=dfy, x="year", y=value_col, ax=ax, showfliers=showfliers)
    ax.set_title(title or f"{value_col} by year")
    return fig, ax


# ----------------------------------------------------------------------
# Correlation (binned x vs. y)
# ----------------------------------------------------------------------


_VALID_CORR_KINDS = ("point", "box", "bar", "violin")


def plot_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    kind: str = "point",
    bin_width: float = 1.0,
    ax=None,
    showfliers: bool = False,
    zero_line: bool = True,
    title: str | None = None,
):
    """Plot the relationship between *x_col* and *y_col* using binned aggregation.

    The x variable (e.g. wind speed, irradiance) is rounded to the nearest
    ``bin_width`` and used as a categorical bin; the y variable (e.g. LMP) is
    aggregated within each bin by seaborn's chosen plot kind.  The Pearson
    correlation coefficient of the raw (unbinned) data is added to the title.

    Args:
        df (pd.DataFrame): Input DataFrame with ``time_utc``, *x_col*, *y_col*.
        x_col (str): X-axis column (e.g. ``"wind_speed"``).
        y_col (str): Y-axis column (e.g. ``"lmp_rt"``).
        kind (str, optional): One of ``"point"``, ``"box"``, ``"bar"``,
            ``"violin"``. Defaults to ``"point"``.
        bin_width (float, optional): Bin width applied to *x_col* before
            aggregation. Defaults to 1.0.
        ax (matplotlib.axes.Axes, optional): Axes to draw on.
        showfliers (bool, optional): Forwarded to ``sns.boxplot`` when
            ``kind="box"``. Defaults to False.
        zero_line (bool, optional): Draw a horizontal ``y=0`` reference line.
            Useful for LMP correlations. Defaults to True.
        title (str, optional): Override title.

    Returns:
        tuple: ``(fig, ax)``.
    """
    _validate_time_utc(df)
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in DataFrame.")
    if kind not in _VALID_CORR_KINDS:
        raise ValueError(f"kind must be one of {_VALID_CORR_KINDS}, got {kind!r}.")
    if bin_width <= 0:
        raise ValueError("bin_width must be > 0.")

    data = df[[x_col, y_col]].dropna().copy()
    if data.empty:
        raise ValueError("No non-NaN rows available for correlation plot.")

    data["__xbin"] = (np.round(data[x_col] / bin_width) * bin_width).astype(float)
    # Format the bin label as int when bin_width is integral, else as float.
    if float(bin_width).is_integer():
        data["__xbin"] = data["__xbin"].astype(int)

    # Pearson r on raw data.
    if data[x_col].std() == 0 or data[y_col].std() == 0:
        pearson_r = float("nan")
    else:
        pearson_r = float(data[x_col].corr(data[y_col]))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    # Sort bin order so categorical plot has a sensible x-axis.
    order = sorted(data["__xbin"].unique())

    if kind == "point":
        sns.pointplot(data=data, x="__xbin", y=y_col, order=order, ax=ax)
    elif kind == "box":
        sns.boxplot(
            data=data,
            x="__xbin",
            y=y_col,
            order=order,
            showfliers=showfliers,
            ax=ax,
        )
    elif kind == "bar":
        sns.barplot(data=data, x="__xbin", y=y_col, order=order, ax=ax)
    else:  # violin
        sns.violinplot(data=data, x="__xbin", y=y_col, order=order, ax=ax)

    if zero_line:
        ax.axhline(0, color="k", linestyle="--", linewidth=1)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(
        title
        if title is not None
        else f"{y_col} vs {x_col} (Pearson r = {pearson_r:.3f})"
    )
    return fig, ax


# ----------------------------------------------------------------------
# Diurnal
# ----------------------------------------------------------------------


def plot_diurnal(
    df: pd.DataFrame,
    value_col: str,
    time_col: str = "time_local",
    agg: str = "mean",
    show_band: bool = True,
    ax=None,
    title: str | None = None,
):
    """Plot the hour-of-day profile of *value_col*.

    Args:
        df (pd.DataFrame): Input DataFrame.
        value_col (str): Column to aggregate.
        time_col (str, optional): Datetime column used for the hour grouping.
            Defaults to ``"time_local"``; falls back to ``"time_utc"`` with a
            warning if missing.
        agg (str, optional): Central-tendency aggregation. One of ``"mean"``
            or ``"median"``. Defaults to ``"mean"``.
        show_band (bool, optional): Shade the inter-quartile range (25–75th
            percentile) around the central line. Defaults to True.
        ax (matplotlib.axes.Axes, optional): Axes to draw on.
        title (str, optional): Axes title.

    Returns:
        tuple: ``(fig, ax)``.
    """
    if time_col not in df.columns:
        if "time_utc" in df.columns:
            import warnings

            warnings.warn(
                f"'{time_col}' not in DataFrame; falling back to 'time_utc'. "
                "Call herc_analysis.utilities.add_local_time(df, lat, lon) "
                "to get a 'time_local' column.",
                stacklevel=2,
            )
            time_col = "time_utc"
        else:
            raise ValueError(
                f"Neither '{time_col}' nor 'time_utc' present in DataFrame."
            )
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not in DataFrame.")
    if agg not in ("mean", "median"):
        raise ValueError("agg must be 'mean' or 'median'.")

    hours = df[time_col].dt.hour
    grouped = df[value_col].groupby(hours)
    center = grouped.mean() if agg == "mean" else grouped.median()
    q25 = grouped.quantile(0.25)
    q75 = grouped.quantile(0.75)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    ax.plot(center.index, center.values, label=agg, color="steelblue")
    if show_band:
        ax.fill_between(
            center.index,
            q25.values,
            q75.values,
            alpha=0.25,
            color="steelblue",
            label="25–75th pct",
        )
    ax.set_xlabel(f"Hour of day ({time_col})")
    ax.set_ylabel(value_col)
    ax.set_xticks(range(0, 24, 3))
    ax.grid(True)
    ax.legend()
    ax.set_title(title or f"Diurnal profile of {value_col}")
    return fig, ax


# ----------------------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------------------


def summary_stats(
    df: pd.DataFrame,
    value_col: str,
    by_year: bool = False,
) -> pd.DataFrame:
    """Compute summary statistics for *value_col*.

    Args:
        df (pd.DataFrame): Input DataFrame with ``time_utc`` and *value_col*.
        value_col (str): Column to summarize.
        by_year (bool, optional): If True, return one row per calendar year.
            Defaults to False (single-row result with index ``"all"``).

    Returns:
        pd.DataFrame: Columns ``count``, ``mean``, ``std``, ``min``,
        ``median``, ``max``, ``pct_negative``.
    """
    _validate_time_utc(df)
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not in DataFrame.")

    def _stats(series: pd.Series) -> dict:
        s = series.dropna()
        if len(s) == 0:
            return {
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "median": np.nan,
                "max": np.nan,
                "pct_negative": np.nan,
            }
        return {
            "count": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "median": float(s.median()),
            "max": float(s.max()),
            "pct_negative": float((s < 0).mean() * 100.0),
        }

    if not by_year:
        return pd.DataFrame([_stats(df[value_col])], index=["all"])

    dfy = _with_year(df)
    years = sorted(dfy["year"].dropna().unique().astype(int))
    rows = {year: _stats(dfy.loc[dfy["year"] == year, value_col]) for year in years}
    return pd.DataFrame.from_dict(rows, orient="index")

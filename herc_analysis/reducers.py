"""Series -> scalar metric reducers. Pure functions.

A reducer collapses a column (or a scalar bundle) to one number, so the metric
definitions read like a list rather than a sprawling method. The guard
conditions mirror the current ``TotalMetrics`` arithmetic exactly.
"""

from __future__ import annotations

import pandas as pd


def total(series: pd.Series) -> float:
    """Sum of a series as a float. Mirrors ``df[col].sum()``.

    Args:
        series (pd.Series): Values to sum.

    Returns:
        float: The sum.
    """
    return float(series.sum())


def mean(series: pd.Series) -> float:
    """Mean of a series as a float. Mirrors ``df[col].mean()``.

    Args:
        series (pd.Series): Values to average.

    Returns:
        float: The mean.
    """
    return float(series.mean())


def capacity_factor(energy_mwh: float, interconnect_mw: float, hours: float) -> float:
    """Plant capacity factor.

    Reproduces the current plant CF:
    ``energy / (interconnect_mw * hours)`` when the denominator is positive,
    else NaN.

    Args:
        energy_mwh (float): Total energy in MWh.
        interconnect_mw (float): Interconnect limit in MW.
        hours (float): Simulation duration in hours.

    Returns:
        float: Capacity factor, or NaN when ``interconnect_mw * hours <= 0``.
    """
    denom = interconnect_mw * hours
    return energy_mwh / denom if denom > 0 else float("nan")


def value_factor(revenue: float, avg_price: float, energy_mwh: float) -> float:
    """Revenue capture relative to flat-price revenue.

    Reproduces the current market value factor: ``revenue / (avg_price *
    energy)`` when both the revenue and the flat-price baseline are positive,
    else NaN.

    Args:
        revenue (float): Realized revenue in dollars.
        avg_price (float): Average price in $/MWh.
        energy_mwh (float): Total energy in MWh.

    Returns:
        float: Value factor, or NaN when ``revenue <= 0`` or the baseline
        ``avg_price * energy <= 0``.
    """
    base = avg_price * energy_mwh
    return revenue / base if revenue > 0 and base > 0 else float("nan")


def tb4_spread(price_hourly: pd.Series, n_pts_4h: int) -> float:
    """Top-4h minus bottom-4h average price (storage arbitrage proxy).

    Reproduces the per-day TB4 lambda in the current
    ``TotalMetrics._compute_tb4_metrics``:
    ``price.nlargest(n).mean() - price.nsmallest(n).mean()``.

    Args:
        price_hourly (pd.Series): Hourly price values for one bucket (e.g. a
            single day).
        n_pts_4h (int): Number of points spanning four hours
            (``int(4 * 3600 / dt_log)``).

    Returns:
        float: The top-minus-bottom average price spread.
    """
    return float(
        price_hourly.nlargest(n_pts_4h).mean() - price_hourly.nsmallest(n_pts_4h).mean()
    )

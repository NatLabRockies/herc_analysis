"""Atomic, vectorized channel builders. No I/O, no object state.

Each function takes pandas Series / scalars and returns a Series (or a small
tuple / scalar). These are the smallest meaningful time-series units and carry
the densest unit tests.

Unit convention (see the refactoring plan, section 6.4):

* power   -- kW  (Hercules convention; raw passthrough, no rebase)
* energy  -- MWh (suffix ``_mwh``)
* price   -- $/MWh
* revenue -- $   (suffix ``_usd``)

Every kW->MWh / kW->MW conversion is confined to one named function below, so a
``$/MWh`` price can only ever meet an ``_mwh`` energy.
"""

from __future__ import annotations

import pandas as pd


def kw_to_mw(power_kw: pd.Series) -> pd.Series:
    """Convert power from kW to MW.

    Used only where power is compared against an MW quantity (e.g. the
    interconnect limit). Mirrors the ``power / 1000.0`` rebase in the current
    ``OutputAnalysis``.

    Args:
        power_kw (pd.Series): Power in kW.

    Returns:
        pd.Series: Power in MW.
    """
    return power_kw / 1000.0


def energy_mwh(power_kw: pd.Series, dt_s: float) -> pd.Series:
    """Incremental energy per logging interval (MWh) from kW power.

    This is the single kW->MWh conversion in the codebase:
    ``kW * s / 3600 / 1000``. It is algebraically identical to the current
    two-step ``power_mw = power_kw / 1000; energy_mwh = power_mw * dt / 3600``.

    Args:
        power_kw (pd.Series): Power in kW.
        dt_s (float): Logging timestep in seconds.

    Returns:
        pd.Series: Incremental energy in MWh for each interval.
    """
    return power_kw * dt_s / 3600.0 / 1000.0


def revenue_usd(price_per_mwh: pd.Series, energy_mwh: pd.Series) -> pd.Series:
    """Revenue in dollars: ``$/MWh * MWh``, elementwise.

    Units line up by construction. Mirrors ``lmp * energy_mwh`` in the current
    component processing.

    Args:
        price_per_mwh (pd.Series): Price in $/MWh.
        energy_mwh (pd.Series): Energy in MWh.

    Returns:
        pd.Series: Revenue in dollars.
    """
    return price_per_mwh * energy_mwh


def charge_discharge_split(energy_mwh: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split signed storage energy into discharge (>=0) and charge (<=0) parts.

    Reproduces the current ``np.where(energy > 0, ...)`` /
    ``np.where(energy < 0, ...)`` storage split: each element appears in exactly
    one of the two outputs, the other carrying ``0.0``.

    Args:
        energy_mwh (pd.Series): Signed storage energy in MWh (positive when
            discharging, negative when charging).

    Returns:
        tuple[pd.Series, pd.Series]: ``(discharge, charge)`` where ``discharge``
        keeps strictly-positive values (else 0.0) and ``charge`` keeps
        strictly-negative values (else 0.0).
    """
    discharge = energy_mwh.where(energy_mwh > 0, 0.0)
    charge = energy_mwh.where(energy_mwh < 0, 0.0)
    return discharge, charge


def soc_mileage(soc: pd.Series) -> float:
    """Total absolute state-of-charge distance travelled (cycling proxy).

    The cumulative absolute step-to-step change in SOC. Mirrors
    ``df[soc_col].diff().abs().sum()`` in the current mileage metric.

    Args:
        soc (pd.Series): State-of-charge time series.

    Returns:
        float: Sum of absolute consecutive SOC differences.
    """
    return float(soc.diff().abs().sum())


def hourly_mean(values: pd.Series, time_utc: pd.Series) -> pd.Series:
    """Broadcast each value to the mean over its clock hour.

    Groups by ``(date, hour)`` of ``time_utc`` and replaces each value with its
    group mean, keeping the original index/length. Mirrors the
    ``lmp_rt_hourly`` construction in the current ``OutputAnalysis``.

    Args:
        values (pd.Series): Values to average (e.g. real-time LMP).
        time_utc (pd.Series): Timezone-aware UTC timestamps aligned to
            ``values``.

    Returns:
        pd.Series: Hourly-mean-broadcast values, same length as ``values``.
    """
    return values.groupby([time_utc.dt.date, time_utc.dt.hour]).transform("mean")

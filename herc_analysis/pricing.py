"""Excess-energy allocation and capacity-headroom pricing. Pure functions.

The two excess-energy corrections in the current ``OutputAnalysis``
(``_apply_excess_charge_correction`` and
``_apply_excess_generation_revenue_correction``) are the same operation: take a
plant-level excess and split it across a set of components in proportion to a
per-row weight. ``allocate_proportional`` is that single primitive; the surplus
and ideal-capacity helpers cover the plant-level headroom pricing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def excess_over_limit(local_gen_mw: pd.Series, interconnect_mw: float) -> pd.Series:
    """Plant generation above the interconnect limit, clipped at zero (MW).

    Mirrors ``(plant_locally_generated_power_mw - interconnect_mw).clip(lower=0)``
    in the current excess-charge correction.

    Args:
        local_gen_mw (pd.Series): Plant locally-generated power in MW.
        interconnect_mw (float): Interconnect limit in MW.

    Returns:
        pd.Series: Non-negative excess local generation in MW.
    """
    return (local_gen_mw - interconnect_mw).clip(lower=0.0)


def allocate_proportional(
    total: pd.Series,
    weights: dict[str, pd.Series],
    *,
    cap_to_weight: bool = False,
) -> dict[str, pd.Series]:
    """Split a per-row total across keys in proportion to per-row weights.

    The single primitive behind BOTH excess corrections in the current
    ``OutputAnalysis``:

    * **generator excess loss** -- ``cap_to_weight=False``, weight is each
      generator's (clipped) power. Reproduces
      ``gen_i * excess / total_gen`` with ``share = (excess / total).fillna(0)``.
    * **storage excess absorption** -- ``cap_to_weight=True``, weight is each
      storage unit's charging power. Reproduces
      ``charge_i * min(1, excess / total_charge)`` with
      ``scale = (excess / total).clip(upper=1).fillna(0)`` so a component never
      absorbs more than it is charging.

    The ``total / Σweights`` ratio uses ``Σweights.replace(0, np.nan)`` followed
    by ``.fillna(0.0)``, matching the current code's zero/NaN edge handling
    exactly (a row with no weight allocates zero to every key).

    Invariants (asserted in tests):

    * uncapped: ``sum(result.values()) == total`` wherever ``Σweights != 0``;
    * every allocation is ``>= 0`` for non-negative inputs;
    * capped: each ``result[k] <= weights[k]``.

    Args:
        total (pd.Series): Per-row total to allocate.
        weights (dict[str, pd.Series]): Per-key, per-row weights.
        cap_to_weight (bool): If True, clip the ratio at 1.0 so each key's
            allocation never exceeds its own weight. Defaults to False.

    Returns:
        dict[str, pd.Series]: Per-key allocation, same keys as ``weights``.
    """
    weight_total = sum(weights.values())
    safe = weight_total.replace(0, np.nan)
    ratio = total / safe
    if cap_to_weight:
        ratio = ratio.clip(upper=1.0)
    ratio = ratio.fillna(0.0)
    return {k: w * ratio for k, w in weights.items()}


def surplus_capacity_mw(interconnect_mw: float, total_power_mw: pd.Series) -> pd.Series:
    """Unused interconnect headroom (MW), signed.

    Mirrors ``interconnect_mw - total_plant_power_mw``. May be negative when the
    plant exceeds the interconnect.

    Args:
        interconnect_mw (float): Interconnect limit in MW.
        total_power_mw (pd.Series): Total plant power in MW.

    Returns:
        pd.Series: Surplus capacity in MW.
    """
    return interconnect_mw - total_power_mw


def split_lmp_sign(lmp: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split a price series into its positive and negative parts.

    Reproduces the current ``np.where(lmp > 0, lmp, 0)`` /
    ``np.where(lmp < 0, lmp, 0)`` split. The negative part retains its sign.

    Args:
        lmp (pd.Series): Locational marginal price.

    Returns:
        tuple[pd.Series, pd.Series]: ``(positive, negative)`` parts.
    """
    return lmp.where(lmp > 0, 0.0), lmp.where(lmp < 0, 0.0)


def capacity_revenue_by_sign(
    capacity_energy_mwh: pd.Series, lmp: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Revenue from a capacity-energy series split by LMP sign.

    Reproduces the current surplus/ideal positive- and negative-LMP revenue
    columns: positive-LMP revenue is ``energy * lmp_pos`` and negative-LMP
    revenue is ``-1 * energy * lmp_neg`` (so it is reported as a positive cost
    avoided / penalty magnitude).

    Args:
        capacity_energy_mwh (pd.Series): Capacity energy in MWh (e.g. surplus
            capacity energy, or interconnect * dt / 3600 for the ideal case).
        lmp (pd.Series): Locational marginal price in $/MWh.

    Returns:
        tuple[pd.Series, pd.Series]: ``(positive_lmp_revenue,
        negative_lmp_revenue)`` in dollars.
    """
    lmp_pos, lmp_neg = split_lmp_sign(lmp)
    return capacity_energy_mwh * lmp_pos, -1.0 * capacity_energy_mwh * lmp_neg

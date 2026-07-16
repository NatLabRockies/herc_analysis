"""Unit + property tests for the L1 pricing/allocation primitives."""

import numpy as np
import pandas as pd

from herc_analysis import pricing


def test_excess_over_limit_clips_at_zero():
    gen = pd.Series([90.0, 100.0, 110.0, 130.0])
    out = pricing.excess_over_limit(gen, interconnect_mw=100.0)
    np.testing.assert_allclose(out, [0.0, 0.0, 10.0, 30.0])


def test_allocation_sums_back_to_total_uncapped():
    total = pd.Series([0.0, 30.0])
    weights = {"a": pd.Series([1.0, 2.0]), "b": pd.Series([1.0, 1.0])}
    alloc = pricing.allocate_proportional(total, weights)
    recombined = sum(alloc.values())
    np.testing.assert_allclose(recombined, total)  # invariant 1
    for s in alloc.values():
        assert (s >= 0).all()  # invariant 2


def test_allocation_proportional_shares():
    # total 30 split across weights 2:1 -> 20 and 10.
    total = pd.Series([30.0])
    weights = {"a": pd.Series([2.0]), "b": pd.Series([1.0])}
    alloc = pricing.allocate_proportional(total, weights)
    np.testing.assert_allclose(alloc["a"], [20.0])
    np.testing.assert_allclose(alloc["b"], [10.0])


def test_zero_weight_row_allocates_zero():
    # No weight anywhere on a row -> every key gets 0 (replace(0,nan)+fillna(0)).
    total = pd.Series([5.0])
    weights = {"a": pd.Series([0.0]), "b": pd.Series([0.0])}
    alloc = pricing.allocate_proportional(total, weights)
    np.testing.assert_allclose(alloc["a"], [0.0])
    np.testing.assert_allclose(alloc["b"], [0.0])


def test_no_op_when_total_is_zero():
    # excess = 0 everywhere -> allocation is identically zero (idempotent no-op).
    total = pd.Series([0.0, 0.0, 0.0])
    weights = {"a": pd.Series([1.0, 2.0, 3.0])}
    alloc = pricing.allocate_proportional(total, weights)
    np.testing.assert_allclose(alloc["a"], [0.0, 0.0, 0.0])


def test_capped_allocation_never_exceeds_weight():
    # More excess (100) than charging weight (5) -> capped at the weight.
    total = pd.Series([100.0])
    weights = {"batt": pd.Series([5.0])}
    alloc = pricing.allocate_proportional(total, weights, cap_to_weight=True)
    np.testing.assert_allclose(alloc["batt"], [5.0])


def test_capped_matches_old_storage_formula():
    # Old: absorbed_i = charge_i * min(1, excess / total_charge).
    excess = pd.Series([3.0, 20.0])
    charge = {"x": pd.Series([2.0, 4.0]), "y": pd.Series([2.0, 6.0])}
    total_charge = charge["x"] + charge["y"]
    scale = (excess / total_charge.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    expected = {k: v * scale for k, v in charge.items()}
    alloc = pricing.allocate_proportional(excess, charge, cap_to_weight=True)
    for k in charge:
        np.testing.assert_allclose(alloc[k], expected[k])
        assert (alloc[k] <= charge[k] + 1e-12).all()


def test_uncapped_matches_old_generator_formula():
    # Old: gen_excess_i = gen_i * excess / total_gen.
    excess = pd.Series([5.0, 0.0, 12.0])
    gen = {"g1": pd.Series([3.0, 1.0, 4.0]), "g2": pd.Series([1.0, 0.0, 8.0])}
    total_gen = gen["g1"] + gen["g2"]
    share = (1.0 / total_gen.replace(0, np.nan)).fillna(0.0)
    expected = {k: v * excess * share for k, v in gen.items()}
    alloc = pricing.allocate_proportional(excess, gen)
    for k in gen:
        np.testing.assert_allclose(alloc[k], expected[k])


def test_surplus_capacity_mw():
    out = pricing.surplus_capacity_mw(22.0, pd.Series([18.0, 22.0, 25.0]))
    np.testing.assert_allclose(out, [4.0, 0.0, -3.0])


def test_split_lmp_sign():
    lmp = pd.Series([10.0, 0.0, -5.0])
    pos, neg = pricing.split_lmp_sign(lmp)
    np.testing.assert_allclose(pos, [10.0, 0.0, 0.0])
    np.testing.assert_allclose(neg, [0.0, 0.0, -5.0])


def test_capacity_revenue_by_sign():
    energy = pd.Series([2.0, 2.0, 2.0])
    lmp = pd.Series([10.0, 0.0, -5.0])
    pos_rev, neg_rev = pricing.capacity_revenue_by_sign(energy, lmp)
    # positive-LMP revenue = energy * lmp_pos; negative reported as magnitude.
    np.testing.assert_allclose(pos_rev, [20.0, 0.0, 0.0])
    np.testing.assert_allclose(neg_rev, [0.0, 0.0, 10.0])

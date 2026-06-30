"""Phase 4 integration tests: the capacity package over the trusted engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from herc_analysis.capacity import MisoCapacity
from herc_analysis.capacity._miso_engine import MisoCapacity as LegacyMisoCapacity
from herc_analysis.comparison import Comparison
from herc_analysis.scenario import Scenario

REPO = Path(__file__).resolve().parents[2]
STORED_05 = (
    REPO
    / "stored_hercules_output"
    / "05_wind_and_storage_with_lmp"
    / "outputs"
    / "hercules_output.h5"
)


def _assert_revenue_dicts_equal(a: dict, b: dict) -> None:
    """Assert two component->revenue dicts match (NaN-aware)."""
    assert a.keys() == b.keys()
    for k in a:
        av, bv = a[k], b[k]
        if isinstance(bv, float) and np.isnan(bv):
            assert np.isnan(av), f"{k}: expected NaN, got {av}"
        else:
            assert av == pytest.approx(bv)


def _frame_48h() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_utc": pd.date_range(
                "2022-09-01 05:00", periods=48, freq="h", tz="UTC"
            ),
            "a": np.linspace(1.0, 4.0, 48) * 1000.0,
            "b": np.linspace(2.0, 5.0, 48) * 1000.0,
        }
    )


def test_wrapper_matches_legacy_engine():
    df = _frame_48h()
    kw = {
        "component_list": ["a", "b"],
        "class_list": ["wind", "wind"],
        "df": df,
        "zone": 1,
        "interconnect_limit": 100_000.0,
        "remove_low_hour_planning_years": False,
    }
    legacy = LegacyMisoCapacity(**kw)
    new = MisoCapacity(**kw)

    _assert_revenue_dicts_equal(new.revenue(), legacy.annual_revenue)
    _assert_revenue_dicts_equal(new.accredit(), legacy.sac_mw)
    # report tables are structurally identical
    new_report = new.report()
    legacy_report = legacy.get_capacity_report()
    pd.testing.assert_frame_equal(new_report["totals"], legacy_report["totals"])


def test_pass_through_to_engine_helpers():
    new = MisoCapacity(
        ["a", "b"],
        ["wind", "wind"],
        _frame_48h(),
        zone=1,
        interconnect_limit=100_000.0,
        remove_low_hour_planning_years=False,
    )
    # An attribute that only exists on the underlying engine resolves via
    # __getattr__ delegation.
    assert new.get_total_revenue() == pytest.approx(sum(new.revenue().values()))


def test_to_metrics_schema_and_total():
    new = MisoCapacity(
        ["a", "b"],
        ["wind", "wind"],
        _frame_48h(),
        zone=1,
        interconnect_limit=100_000.0,
        sim_years=1.0,
        remove_low_hour_planning_years=False,
    )
    m = new.to_metrics()
    assert {"entity", "metric", "value", "scaling", "sim_years"}.issubset(m.columns)
    assert (m["metric"] == "capacity_revenue").all()
    assert (m["scaling"] == "annual").all()
    plant = m.loc[m["entity"] == "plant", "value"].iloc[0]
    comps = m.loc[m["entity"] != "plant", "value"].sum()
    assert plant == pytest.approx(comps)


def test_from_scenario_matches_legacy_built_from_same_inputs():
    s = Scenario(str(STORED_05), name="wind_storage")
    new = MisoCapacity.from_scenario(s, zone=1, remove_low_hour_planning_years=False)

    # Build the legacy engine from the same scenario-derived inputs.
    ch = s.channels
    df = pd.DataFrame({"time_utc": ch["time_utc"].reset_index(drop=True)})
    for name in new.components:
        df[name] = ch[f"{name}__power_kw"].reset_index(drop=True)
    legacy = LegacyMisoCapacity(
        component_list=new.components,
        class_list=new.classes,
        df=df,
        zone=1,
        interconnect_limit=s.meta.interconnect_mw * 1000.0,
        remove_low_hour_planning_years=False,
    )
    _assert_revenue_dicts_equal(new.revenue(), legacy.annual_revenue)


def test_capacity_metrics_compose_into_comparison():
    # Use the synthetic 48-hour case, which yields finite revenue.
    cap = MisoCapacity(
        ["a", "b"],
        ["wind", "wind"],
        _frame_48h(),
        zone=1,
        interconnect_limit=100_000.0,
        sim_years=1.0,
        remove_low_hour_planning_years=False,
    )
    long = cap.to_metrics()
    long["case"] = "synthetic"
    cmp = Comparison(long)
    annual = cmp.table("capacity_revenue", view="per_year", entity="plant")
    # scaling="annual" -> annual view is the raw value, unchanged.
    assert annual.loc["synthetic", "capacity_revenue"] == pytest.approx(
        sum(cap.revenue().values())
    )


def test_from_scenario_requires_mappable_component():
    s = Scenario(str(STORED_05), name="wind_storage")
    # Sanity: stored_05 has at least one MISO-mappable component, so this works.
    cap = MisoCapacity.from_scenario(s, zone=1, remove_low_hour_planning_years=False)
    assert cap.components

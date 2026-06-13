"""Tests for the ScenarioComparison cross-scenario engine."""

import numpy as np
import pandas as pd
import pytest

from herc_analysis import ScenarioComparison


def _make_long_metrics() -> pd.DataFrame:
    """Build a small two-case long-format metrics frame for testing.

    Returns:
        pd.DataFrame: Long-format metrics for cases ``A`` (2 years) and ``B``
            (4 years).
    """
    rows = [
        # case A: 2-year simulation
        ("A", "plant", "energy_mwh", 200.0, "MWh", "extensive", 2.0),
        ("A", "plant", "capacity_factor", 0.5, "-", "intensive", 2.0),
        ("A", "battery", "capacity_revenue", 10.0, "M$", "annual", 2.0),
        ("A", "battery", "battery_mileage_soc", 8.0, "SOC", "extensive", 2.0),
        # case B: 4-year simulation
        ("B", "plant", "energy_mwh", 800.0, "MWh", "extensive", 4.0),
        ("B", "plant", "capacity_factor", 0.6, "-", "intensive", 4.0),
        ("B", "battery", "capacity_revenue", 12.0, "M$", "annual", 4.0),
        ("B", "battery", "battery_mileage_soc", 40.0, "SOC", "extensive", 4.0),
    ]
    return pd.DataFrame(
        rows,
        columns=["case", "scope", "metric", "value", "unit", "scaling", "sim_years"],
    )


def test_compare_annual_scales_extensive_by_sim_years():
    """Annual period divides extensive metrics by sim_years; ratios unchanged."""
    sc = ScenarioComparison(_make_long_metrics(), case_names=["A", "B"])
    wide = sc.compare(["energy_mwh", "capacity_factor"], period="annual", scope="plant")

    np.testing.assert_allclose(wide.loc["A", "energy_mwh"], 100.0)
    np.testing.assert_allclose(wide.loc["B", "energy_mwh"], 200.0)
    # Intensive metric is left untouched.
    np.testing.assert_allclose(wide.loc["A", "capacity_factor"], 0.5)
    np.testing.assert_allclose(wide.loc["B", "capacity_factor"], 0.6)


def test_compare_total_leaves_extensive_and_scales_annual():
    """Total period keeps extensive values and multiplies annual ones."""
    sc = ScenarioComparison(_make_long_metrics(), case_names=["A", "B"])

    wide_energy = sc.compare("energy_mwh", period="total", scope="plant")
    np.testing.assert_allclose(wide_energy.loc["A", "energy_mwh"], 200.0)
    np.testing.assert_allclose(wide_energy.loc["B", "energy_mwh"], 800.0)

    wide_cap = sc.compare("capacity_revenue", period="total", scope="battery")
    # annual -> total multiplies by sim_years.
    np.testing.assert_allclose(wide_cap.loc["A", "capacity_revenue"], 20.0)
    np.testing.assert_allclose(wide_cap.loc["B", "capacity_revenue"], 48.0)


def test_compare_relabels_with_display_names():
    """display_names replace case names as the comparison row index."""
    sc = ScenarioComparison(
        _make_long_metrics(),
        case_names=["A", "B"],
        display_names=["Case Alpha", "Case Beta"],
    )
    wide = sc.compare("energy_mwh", period="annual", scope="plant")

    assert list(wide.index) == ["Case Alpha", "Case Beta"]
    np.testing.assert_allclose(wide.loc["Case Alpha", "energy_mwh"], 100.0)


def test_display_names_length_mismatch_raises():
    """A display_names list that does not match case_names raises ValueError."""
    with pytest.raises(ValueError, match="display_names length"):
        ScenarioComparison(
            _make_long_metrics(), case_names=["A", "B"], display_names=["only_one"]
        )


def test_compare_missing_scope_returns_nan():
    """Comparing a scope absent from every case yields an all-NaN frame."""
    sc = ScenarioComparison(_make_long_metrics(), case_names=["A", "B"])
    wide = sc.compare(["energy_mwh"], period="annual", scope="nonexistent")

    assert list(wide.index) == ["A", "B"]
    assert list(wide.columns) == ["energy_mwh"]
    assert wide["energy_mwh"].isna().all()


def test_compare_all_scopes_uses_multiindex_columns():
    """scope='all' yields a (scope, metric) column MultiIndex."""
    sc = ScenarioComparison(_make_long_metrics(), case_names=["A", "B"])
    wide = sc.compare("capacity_revenue", period="annual", scope="all")
    assert isinstance(wide.columns, pd.MultiIndex)
    np.testing.assert_allclose(wide.loc["A", ("battery", "capacity_revenue")], 10.0)


def test_init_requires_schema_columns():
    """A frame missing required columns raises ValueError."""
    bad = pd.DataFrame({"case": ["A"], "metric": ["energy_mwh"], "value": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        ScenarioComparison(bad)


def test_collect_reads_and_tags_cases(tmp_path):
    """collect() reads per-case files and tags rows with case names."""
    metrics = _make_long_metrics()
    for case in ("case1", "case2"):
        out = tmp_path / case / "outputs"
        out.mkdir(parents=True)
        metrics.drop(columns="case").iloc[:4].to_csv(out / "metrics.csv", index=False)

    long_df = ScenarioComparison.collect(
        [tmp_path / "case1", tmp_path / "case2"],
    )
    assert set(long_df["case"]) == {"case1", "case2"}
    assert len(long_df) == 8


def test_run_analysis_in_cases_executes_script(tmp_path):
    """run_analysis_in_cases copies a script into each case and runs it."""
    script = tmp_path / "compute_metrics.py"
    script.write_text(
        "from pathlib import Path\n"
        "Path('outputs').mkdir(exist_ok=True)\n"
        "Path('outputs/ran.txt').write_text('ok')\n"
    )
    case_dir = tmp_path / "caseX"
    case_dir.mkdir()

    results = ScenarioComparison.run_analysis_in_cases([case_dir], script)
    assert results[0][1] == 0
    assert (case_dir / "outputs" / "ran.txt").read_text() == "ok"

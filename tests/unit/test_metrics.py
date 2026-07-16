"""Unit tests for the long-format MetricSet value object."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from herc_analysis.metrics import METRIC_COLUMNS, MetricSet, MetricSpec


def _sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entity": "plant",
                "metric": "energy_mwh",
                "resolution": "total",
                "period": "total",
                "value": 100.0,
                "unit": "MWh",
                "scaling": "extensive",
            },
            {
                "entity": "plant",
                "metric": "energy_mwh",
                "resolution": "annual",
                "period": "annual",
                "value": 50.0,
                "unit": "MWh",
                "scaling": "extensive",
            },
            {
                "entity": "plant",
                "metric": "capacity_factor",
                "resolution": "total",
                "period": "total",
                "value": 0.5,
                "unit": "-",
                "scaling": "intensive",
            },
            {
                "entity": "battery",
                "metric": "energy_mwh",
                "resolution": "monthly",
                "period": "2024-01",
                "value": 7.0,
                "unit": "MWh",
                "scaling": "extensive",
            },
        ]
    )


def test_metric_spec_is_frozen_record():
    spec = MetricSpec(name="energy_mwh", unit="MWh", scaling="extensive")
    assert (spec.name, spec.unit, spec.scaling) == ("energy_mwh", "MWh", "extensive")


def test_requires_columns():
    bad = pd.DataFrame({"entity": ["plant"], "metric": ["x"]})
    try:
        MetricSet(bad, sim_years=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing columns")


def test_scalar_lookup_and_missing():
    ms = MetricSet(_sample_rows(), sim_years=2.0)
    assert ms.scalar("plant", "energy_mwh") == 100.0
    assert (
        ms.scalar("battery", "energy_mwh", resolution="monthly", period="2024-01")
        == 7.0
    )
    assert math.isnan(ms.scalar("plant", "nonexistent"))


def test_scalar_annual_defaults_to_annual_period():
    ms = MetricSet(_sample_rows(), sim_years=2.0)
    assert ms.scalar("plant", "energy_mwh", resolution="annual") == 50.0


def test_scalar_calendar_resolution_requires_period():
    ms = MetricSet(_sample_rows(), sim_years=2.0)
    with pytest.raises(ValueError, match="period is required"):
        ms.scalar("battery", "energy_mwh", resolution="monthly")


def test_at_resolution():
    ms = MetricSet(_sample_rows(), sim_years=2.0)
    total = ms.at("total")
    assert set(total["entity"]) == {"plant"}
    assert len(ms.at("monthly")) == 1


def test_to_long_adds_sim_years_and_keeps_columns():
    ms = MetricSet(_sample_rows(), sim_years=2.0)
    long = ms.to_long()
    assert list(long.columns) == [*METRIC_COLUMNS, "sim_years"]
    assert (long["sim_years"] == 2.0).all()
    assert list(ms.to_long(with_sim_years=False).columns) == list(METRIC_COLUMNS)


def test_to_csv_round_trips_via_from_long(tmp_path):
    ms = MetricSet(_sample_rows(), sim_years=2.0)
    path = ms.to_csv(tmp_path / "metrics.csv")
    reloaded = MetricSet.from_long(pd.read_csv(path))
    assert reloaded.sim_years == 2.0
    assert reloaded.scalar("plant", "energy_mwh") == 100.0


def test_to_nested_total():
    ms = MetricSet(_sample_rows(), sim_years=2.0)
    nested = ms.to_nested()
    assert nested["plant"]["energy_mwh"] == 100.0
    assert nested["plant"]["capacity_factor"] == 0.5
    assert "battery" not in nested  # battery row is monthly, not total


def test_from_long_requires_sim_years():
    rows = _sample_rows()  # no sim_years column
    try:
        MetricSet.from_long(rows)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError without sim_years")

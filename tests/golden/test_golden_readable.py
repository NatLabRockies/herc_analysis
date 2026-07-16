"""Phase-0-safe checks that the frozen golden snapshots are present and parseable.

These do not recompute anything -- the real parity tests that rebuild outputs
with ``Scenario`` and compare against these snapshots land in Phase 2. Here we
only guard that the committed contract files exist and load cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parent / "data"

EXPECTED_DATASETS = ("fixture", "stored_05", "stored_02b")


def test_data_dir_exists():
    assert DATA_DIR.is_dir(), f"golden data dir missing: {DATA_DIR}"


@pytest.mark.parametrize("name", EXPECTED_DATASETS)
def test_metrics_json_parses(name):
    path = DATA_DIR / f"{name}.metrics.json"
    assert path.exists(), f"missing golden metrics: {path}"
    data = json.loads(path.read_text())
    assert isinstance(data, dict) and data, "metrics JSON is empty"
    assert "plant" in data, "metrics JSON missing 'plant' section"


@pytest.mark.parametrize("name", EXPECTED_DATASETS)
def test_channels_snapshot_parses(name):
    parquet = DATA_DIR / f"{name}.channels.parquet"
    summary = DATA_DIR / f"{name}.channels_summary.json"
    assert parquet.exists() or summary.exists(), f"missing golden channels for {name}"
    if parquet.exists():
        df = pd.read_parquet(parquet)
        assert len(df) > 0 and len(df.columns) > 0
    else:
        data = json.loads(summary.read_text())
        assert data["columns"] and data["n_rows"] > 0


def test_fixture_known_capacity_factor():
    """The deterministic fixture pins the trusted plant CF = 18 / 22."""
    data = json.loads((DATA_DIR / "fixture.metrics.json").read_text())
    assert data["plant"]["capacity_factor"] == pytest.approx(18 / 22)

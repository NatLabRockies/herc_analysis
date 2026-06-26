"""Freeze the current code's outputs as golden parity snapshots.

Phase 0 of the refactor. Builds, from the *current* ``OutputAnalysis`` +
``TotalMetrics``, the reference outputs that later phases must reproduce
numerically (the "no behavior change" contract):

* ``<name>.metrics.json``  -- ``TotalMetrics.compute_metrics``
* ``<name>.monthly.json``  -- ``TotalMetrics.compute_monthly_metrics``
* ``<name>.channels.parquet``      -- the derived ``OutputAnalysis.df`` (full
  frame), when small enough; otherwise
* ``<name>.channels_summary.json`` -- column list + per-column
  ``(sum, mean, min, max)`` when the full frame would be large.

Run from the repo root::

    uv run python tests/golden/generate_golden.py

Regeneration is deterministic: re-running produces byte-identical files.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing the deterministic H5 fixture builder from tests/conftest.py.
_TESTS_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _TESTS_DIR.parent
sys.path.insert(0, str(_TESTS_DIR))

from conftest import _create_test_h5  # noqa: E402

from herc_analysis import OutputAnalysis, TotalMetrics  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"

# Store the full derived frame as parquet when it is at most this many bytes;
# otherwise fall back to a per-column statistical summary to keep the repo lean.
_MAX_PARQUET_BYTES = 5 * 1024 * 1024

_STORED = _REPO_ROOT / "stored_hercules_output"
_REAL_DATASETS = {
    "stored_05": _STORED
    / "05_wind_and_storage_with_lmp"
    / "outputs"
    / "hercules_output.h5",
    "stored_02b": _STORED
    / "02b_wind_farm_realistic_inflow_precom_floris"
    / "outputs"
    / "hercules_output.h5",
}


def _json_default(obj):
    """Convert numpy / pandas scalars to JSON-native types, exactly.

    Floats are emitted via ``repr`` round-tripping (json handles Python floats
    losslessly); NaN/inf are preserved by the json module. Anything still
    unknown falls back to ``str`` so the snapshot never fails to write.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def _write_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def _freeze_channels(name: str, df: pd.DataFrame) -> None:
    """Freeze the derived channels frame, full or summarized by size."""
    parquet_path = DATA_DIR / f"{name}.channels.parquet"
    df.to_parquet(parquet_path, index=False)
    if parquet_path.stat().st_size > _MAX_PARQUET_BYTES:
        parquet_path.unlink()
        numeric = df.select_dtypes(include=[np.number])
        summary = {
            "n_rows": int(len(df)),
            "columns": list(df.columns),
            "stats": {
                col: {
                    "sum": float(numeric[col].sum()),
                    "mean": float(numeric[col].mean()),
                    "min": float(numeric[col].min()),
                    "max": float(numeric[col].max()),
                }
                for col in numeric.columns
            },
        }
        _write_json(DATA_DIR / f"{name}.channels_summary.json", summary)
        print(f"  channels: summary ({len(df.columns)} cols, {len(df)} rows)")
    else:
        print(f"  channels: parquet ({parquet_path.stat().st_size} bytes)")


def _freeze_dataset(name: str, h5_path: Path) -> None:
    print(f"[{name}] {h5_path}")
    oa = OutputAnalysis(str(h5_path))
    tm = TotalMetrics(oa)

    metrics = tm.compute_metrics(display=False)
    _write_json(DATA_DIR / f"{name}.metrics.json", metrics)
    print("  metrics: written")

    try:
        monthly = tm.compute_monthly_metrics(display=False)
        _write_json(DATA_DIR / f"{name}.monthly.json", monthly)
        print("  monthly: written")
    except ValueError as exc:
        print(f"  monthly: skipped ({exc})")

    _freeze_channels(name, oa.df)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Deterministic fixture (tiny, exact known values like CF = 18/22).
    with tempfile.TemporaryDirectory() as tmp:
        fixture_h5 = Path(tmp) / "hercules_output.h5"
        _create_test_h5(str(fixture_h5))
        _freeze_dataset("fixture", fixture_h5)

    # Real bundled runs, when present.
    for name, path in _REAL_DATASETS.items():
        if path.exists():
            _freeze_dataset(name, path)
        else:
            print(f"[{name}] missing, skipped: {path}")


if __name__ == "__main__":
    main()

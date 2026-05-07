"""Convert the RA hour XLSX sheets provided by MISO into a single csv file.

All ``*.xlsx`` files in this directory are read from their ``DATA`` tab and
stitched together in time.  ``time_utc`` marks the START of each hour.

The output csv file and a log file are created and these are tracked in the git repository.

To generate a new csv file using update data,
 add spreadsheets of RA data (can see log for example files) and run this script.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

INPUT_DIR = Path(__file__).parent
OUTPUT_PATH = INPUT_DIR / "miso_ra_hours.csv"
LOG_PATH = INPUT_DIR / "miso_ra_hours_log.txt"

SEASONAL_CENTRAL_NORTH = "Seasonal RA Hour_RA Hour Identifier (Central + North)"
SEASONAL_SOUTH = "Seasonal RA Hour_RA Hour Identifier (South)"
AAOC_CENTRAL_NORTH = "Annual RA Hour (AAOC Hour)_RA Hour Identifier (Central + North)"
AAOC_SOUTH = "Annual RA Hour (AAOC Hour)_RA Hour Identifier (South)"


ONE_HOUR = pd.Timedelta(hours=1)


def _flatten_columns(cols):
    """Join the two header rows, dropping placeholder ``Unnamed:*`` levels."""
    return [
        "_".join(s for s in (str(x) for x in c) if not s.startswith("Unnamed")).strip(
            "_"
        )
        for c in cols
    ]


def _load_one(path):
    """Load one MISO RA-hour xlsx into a tidy per-file DataFrame."""
    raw = pd.read_excel(path, sheet_name="DATA", header=[0, 1])
    raw.columns = _flatten_columns(raw.columns)

    # timeest is hour-beginning EST with NO daylight-savings shifts (verified
    # empirically: every lookback year has perfectly uniform 1-hour spacing,
    # including leap years giving exactly 8784 rows).  Localize to fixed
    # UTC-5 (POSIX sign convention => "Etc/GMT+5") and convert to UTC.
    time_utc = raw["timeest"].dt.tz_localize("Etc/GMT+5").dt.tz_convert("UTC")

    out = pd.DataFrame(
        {
            "time_utc": time_utc,
            "season": raw["season"].astype("string"),
            # "lookback_year": raw["CY: LookBackYear"].astype("string"),
            "ra_central_north": raw[SEASONAL_CENTRAL_NORTH].astype(int),
            "ra_south": raw[SEASONAL_SOUTH].astype(int),
        }
    )
    if AAOC_CENTRAL_NORTH in raw.columns:
        out["aaoc_central_north"] = raw[AAOC_CENTRAL_NORTH].astype(int)
        out["aaoc_south"] = raw[AAOC_SOUTH].astype(int)

    diffs = out["time_utc"].sort_values().diff().dropna().unique()
    assert len(diffs) == 1 and diffs[0] == ONE_HOUR, (
        f"{path.name}: irregular hour spacing (possible DST shift): {diffs}"
    )
    return out


def main():
    files = sorted(p for p in INPUT_DIR.glob("*.xlsx") if not p.name.startswith("~$"))
    assert files, f"No .xlsx files found in {INPUT_DIR}"

    parts = [_load_one(f) for f in files]
    expected_rows = sum(len(p) for p in parts)

    df = (
        pd.concat(parts, ignore_index=True)
        .sort_values("time_utc")
        .reset_index(drop=True)
    )

    assert len(df) == expected_rows, f"row count mismatch: {len(df)} vs {expected_rows}"
    n_dupes = int(df["time_utc"].duplicated().sum())
    gap_diffs = df["time_utc"].diff().dropna()
    n_gaps = int((gap_diffs != ONE_HOUR).sum())

    print("Head:")
    print(df.head())
    print("\nTail:")
    print(df.tail())
    print(f"\nFiles read ({len(files)}):")
    for f in files:
        print(f"  {f.name}")
    print(f"\nRows: {len(df)} (expected {expected_rows})")
    print(f"Time range: {df['time_utc'].min()}  ->  {df['time_utc'].max()}")
    print(f"Duplicate timestamps: {n_dupes}")
    print(f"Gaps (non-1h diffs): {n_gaps}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    # Log the names of the files read and the date this was run into a log file
    with open(LOG_PATH, "a") as f:
        f.write(f"Files read: {', '.join([f.name for f in files])}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Time range: {df['time_utc'].min()}  ->  {df['time_utc'].max()}\n")
        f.write(f"Duplicate timestamps: {n_dupes}\n")
        f.write(f"Gaps (non-1h diffs): {n_gaps}\n")
        f.write(f"Columns: {list(df.columns)}\n")
        f.write(f"Dtypes:\n{df.dtypes}\n")
        f.write("========================================\n")


if __name__ == "__main__":
    main()

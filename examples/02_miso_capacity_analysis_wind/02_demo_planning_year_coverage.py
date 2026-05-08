"""Example 02 — Visualize planning-year coverage of a Hercules simulation.

MISO planning years run September 1 (UTC) of year Y through August 31
(UTC) of year Y+1.  The bundled simulation spans roughly Aug 2023 → Sep
2024, which fully covers planning year 2324 plus partial fringes of
PY 2223 and PY 2425.  ``plot_planning_year_coverage`` draws each data
column with complete planning years in black and incomplete ones in
pale red, with a labeled arrow above each complete window.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from herc_analysis import OutputAnalysis
from herc_analysis.miso_capacity import plot_planning_year_coverage

DATA_DIR = Path(__file__).parent

oa = OutputAnalysis(DATA_DIR / "hercules_output.h5")
df_raw = oa.df[
    [
        "time_utc",
        "wind_farm.power",
        "external_signals.lmp_rt",
        "external_signals.lmp_da",
    ]
].copy()

plot_planning_year_coverage(df_raw)
plt.show()

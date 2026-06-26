"""Example 02 — Wind-only MISO capacity: seasonal availability and ISAC.

Builds a MisoCapacity for the wind farm at the real interconnect limit and
produces two figures:

1. Bar chart comparing Tier 1, Tier 2, and ISAC availability by season.
2. Scatter plots of hourly wind output by season with RA hours highlighted,
   overlaid with horizontal lines for the three availability metrics.

The tier-1 / tier-2 split and the ISAC formula are the core of MISO's
accredited capacity methodology:

    ISAC = 0.2 * Tier1  +  0.8 * Tier2

where Tier 2 covers the RA (Resource Adequacy) hours and Tier 1 covers all
other classified hours.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from herc_analysis import Scenario
from herc_analysis.capacity import MisoCapacity

DATA_DIR = Path(__file__).parent
ZONE = 1
INTERCONNECT_LIMIT_KW = 61_500.0  # kW

oa = Scenario(DATA_DIR / "hercules_output.h5")
df_raw = oa.output.df[["time_utc", "wind_farm.power"]].copy()

mc = MisoCapacity(
    component_list=["wind_farm.power"],
    class_list=["wind"],
    df=df_raw,
    zone=ZONE,
    interconnect_limit=INTERCONNECT_LIMIT_KW,
    verbose=True,
    remove_low_hour_planning_years=True,
)

COMPONENT = "wind_farm.power"

# ── Figure 1: seasonal availability bar chart ────────────────────────────────
fig1 = mc.plot_isac_availability_bar(COMPONENT)

# ── Figure 2: hourly scatter by season with RA-hour overlay ─────────────────
fig2 = mc.plot_isac_hourly_scatter(COMPONENT)

plt.show()

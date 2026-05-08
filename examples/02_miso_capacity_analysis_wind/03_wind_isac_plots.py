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
import numpy as np

from herc_analysis import OutputAnalysis
from herc_analysis.miso_capacity import MisoCapacity

DATA_DIR = Path(__file__).parent
ZONE = 1
INTERCONNECT_LIMIT_KW = 61_500.0  # kW

oa = OutputAnalysis(DATA_DIR / "hercules_output.h5")
df_raw = oa.df[["time_utc", "wind_farm.power"]].copy()

mc = MisoCapacity(
    component_list=["wind_farm.power"],
    class_list=["wind"],
    df=df_raw,
    zone=ZONE,
    interconnect_limit=INTERCONNECT_LIMIT_KW,
)

COMPONENT = "wind_farm.power"
seasons = [
    s for s in ("summer", "fall", "winter", "spring") if (s, COMPONENT) in mc.isac_mw
]

tier1 = [mc.tier_1_availability_mw[(s, COMPONENT)] for s in seasons]
tier2 = [mc.tier_2_availability_mw[(s, COMPONENT)] for s in seasons]
isac = [mc.isac_mw[(s, COMPONENT)] for s in seasons]

# AAOC: annual value (not seasonal), use the first (planning_year, component) entry
aaoc_value = list(mc.aaoc_per_planning_year_mw.values())[0]

# Count Tier 1 (non-RA) and Tier 2 (RA) hours per season
ra_col = f"ra_{mc.subregion}"
df_h = mc.df_h_limit_mw
tier1_hours = {}
tier2_hours = {}
for s in seasons:
    df_s = df_h[df_h["season"] == s]
    tier2_hours[s] = int(df_s[ra_col].sum())
    tier1_hours[s] = len(df_s) - tier2_hours[s]

# ── Figure 1: seasonal availability bar chart ────────────────────────────────
x = np.arange(len(seasons))
width = 0.25

fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.bar(x - width, tier1, width, label="Tier 1 (non-RA hours)", color="steelblue")
ax1.bar(x, tier2, width, label="Tier 2 (RA hours)", color="darkorange")
ax1.bar(x + width, isac, width, label="ISAC (0.2·T1 + 0.8·T2)", color="seagreen")
ax1.axhline(
    aaoc_value, color="purple", ls="--", lw=1.5, label=f"AAOC ({aaoc_value:.1f} MW)"
)
ax1.set_xticks(x)
ax1.set_xticklabels([s.capitalize() for s in seasons])
ax1.set_ylabel("Availability (MW)")
ax1.set_title("Wind farm — seasonal Tier 1 / Tier 2 / ISAC availability")
ax1.legend()
ax1.grid(axis="y")
plt.tight_layout()

# ── Figure 2: hourly scatter by season with RA-hour overlay ─────────────────
ra_col = f"ra_{mc.subregion}"
df_h = mc.df_h_limit_mw

fig2, axes2 = plt.subplots(1, len(seasons), figsize=(5 * len(seasons), 5), sharey=True)

season_color = {
    "summer": "tab:orange",
    "fall": "tab:brown",
    "winter": "tab:blue",
    "spring": "tab:green",
}

for ax, season in zip(axes2, seasons, strict=False):
    df_s = df_h[df_h["season"] == season].reset_index(drop=True)
    ra_mask = df_s[ra_col]

    ax.scatter(
        df_s.index[~ra_mask],
        df_s.loc[~ra_mask, COMPONENT],
        s=2,
        alpha=0.3,
        color="steelblue",
        label=f"Non-RA hours (Tier 1: {tier1_hours[season]}h)",
    )
    ax.scatter(
        df_s.index[ra_mask],
        df_s.loc[ra_mask, COMPONENT],
        s=6,
        alpha=0.7,
        color="tomato",
        label=f"RA hours (Tier 2: {tier2_hours[season]}h)",
    )

    key = (season, COMPONENT)
    ax.axhline(
        mc.tier_1_availability_mw[key],
        color="steelblue",
        ls="--",
        lw=1.5,
        label=f"Tier 1 ({mc.tier_1_availability_mw[key]:.1f} MW)",
    )
    ax.axhline(
        mc.tier_2_availability_mw[key],
        color="darkorange",
        ls="--",
        lw=1.5,
        label=f"Tier 2 ({mc.tier_2_availability_mw[key]:.1f} MW)",
    )
    ax.axhline(
        mc.isac_mw[key],
        color="seagreen",
        ls="-",
        lw=2,
        label=f"ISAC  ({mc.isac_mw[key]:.1f} MW)",
    )
    ax.axhline(
        aaoc_value,
        color="purple",
        ls="--",
        lw=1.5,
        label=f"AAOC  ({aaoc_value:.1f} MW)",
    )

    ax.set_title(season.capitalize())
    ax.set_xlabel("Hour index (within season)")
    if ax is axes2[0]:
        ax.set_ylabel("Wind Power (MW)")
    ax.legend(markerscale=4, fontsize=8)

fig2.suptitle("Hourly wind output — RA hours highlighted with ISAC metrics")
plt.tight_layout()
plt.show()

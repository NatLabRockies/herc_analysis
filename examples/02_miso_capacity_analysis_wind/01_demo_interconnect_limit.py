"""Example 01 — Interconnect limit: how MisoCapacity caps hourly wind output.

MisoCapacity automatically caps each hour's component power to the interconnect
limit before computing availability metrics.  This example makes the capping
visible by running the same wind data through two different limits:

- Real limit: 61.5 MW (matches the rated plant capacity, so little capping)
- Tight limit: 30 MW  (many hours capped, making the effect obvious)

The before/after comparison is plotted for January.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from herc_analysis import OutputAnalysis
from herc_analysis.miso_capacity import MisoCapacity

DATA_DIR = Path(__file__).parent
ZONE = 1
REAL_LIMIT_KW = 61_500.0  # actual plant interconnect (kW)
TIGHT_LIMIT_KW = 30_000.0  # reduced for illustration (kW)

oa = OutputAnalysis(DATA_DIR / "hercules_output.h5")
df_raw = oa.df[["time_utc", "wind_farm.power"]].copy()

# Build two MisoCapacity objects — one at each limit.
# remove_low_hour_seasons=False because the simulation is a partial year and
# winter falls below the 85-day threshold.
mc_real = MisoCapacity(
    component_list=["wind_farm.power"],
    class_list=["wind"],
    df=df_raw,
    zone=ZONE,
    interconnect_limit=REAL_LIMIT_KW,
    remove_low_hour_seasons=False,
)

mc_tight = MisoCapacity(
    component_list=["wind_farm.power"],
    class_list=["wind"],
    df=df_raw,
    zone=ZONE,
    interconnect_limit=TIGHT_LIMIT_KW,
    remove_low_hour_seasons=False,
)

# df_h_mw     — hourly data before any capping (MW, averaged to the hour)
# df_h_limit_mw — hourly data after capping to the interconnect limit (MW)
df_before = mc_tight.df_h_mw
df_after = mc_tight.df_h_limit_mw

jan_before = df_before[df_before["time_utc"].dt.month == 1]
jan_after = df_after[df_after["time_utc"].dt.month == 1]

# ── Figure 1: real limit — capping is largely invisible ─────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

ax = axes[0]
df_real_before = mc_real.df_h_mw
df_real_after = mc_real.df_h_limit_mw
jan_rb = df_real_before[df_real_before["time_utc"].dt.month == 1]
jan_ra = df_real_after[df_real_after["time_utc"].dt.month == 1]
ax.plot(
    jan_rb["time_utc"],
    jan_rb["wind_farm.power"],
    color="steelblue",
    lw=1,
    label="Uncapped",
)
ax.plot(
    jan_ra["time_utc"],
    jan_ra["wind_farm.power"],
    color="tomato",
    lw=1,
    ls="--",
    label="Capped (hidden behind uncapped)",
)
ax.axhline(
    REAL_LIMIT_KW / 1000.0,
    color="green",
    ls="--",
    lw=1.5,
    label=f"Interconnect limit ({REAL_LIMIT_KW / 1000:.0f} MW)",
)
ax.set_ylabel("Wind Power (MW)")
ax.set_title(
    f"Real interconnect limit — {REAL_LIMIT_KW / 1000:.0f} MW "
    f"(wind rarely touches the cap)"
)
ax.legend(fontsize=9)
ax.grid(True)

# ── Figure 2: tight limit — capping is obvious ───────────────────────────────
ax = axes[1]
ax.plot(
    jan_before["time_utc"],
    jan_before["wind_farm.power"],
    color="steelblue",
    lw=1,
    alpha=0.7,
    label="Uncapped",
)
ax.plot(
    jan_after["time_utc"],
    jan_after["wind_farm.power"],
    color="tomato",
    lw=1,
    label="Capped",
)
ax.axhline(
    TIGHT_LIMIT_KW / 1000.0,
    color="red",
    ls="--",
    lw=1.5,
    label=f"Interconnect limit ({TIGHT_LIMIT_KW / 1000:.0f} MW)",
)
ax.set_ylabel("Wind Power (MW)")
ax.set_xlabel("Time (UTC)")
ax.set_title(
    f"Tight interconnect limit — {TIGHT_LIMIT_KW / 1000:.0f} MW (many hours capped)"
)
ax.legend(fontsize=9)
ax.grid(True)

plt.tight_layout()
plt.show()

"""Example 07 — Stacked bar plots of SAC and capacity revenue.

Reuses the wind + battery configuration from example 06 and demonstrates
the new ``MisoCapacity.plot_sac_stacked_bar`` and
``MisoCapacity.plot_revenue_stacked_bar`` methods.  Each method renders a
per-season bar (summer / fall / winter / spring) whose total height is
annotated and whose stacks are colored by component.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from herc_analysis import Scenario
from herc_analysis.capacity import MisoCapacity, compute_battery_availability

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ZONE = 1
INTERCONNECT_LIMIT_KW = 61_500.0  # kW

BATTERY_RATED_POWER_KW = 20_000.0  # kW  (20 MW)
BATTERY_RATED_ENERGY_KWH = 80_000.0  # kWh (80 MWh — 4-hour duration)
BATTERY_MIN_SOC = 0.0
ETA_DISCHARGE = 1.0

oa = Scenario(DATA_DIR / "hercules_output.h5")
df = oa.output.df[["time_utc", "wind_farm.power"]].copy()

# Add an idle, fully-charged battery as a dedicated capacity resource.
df["battery_power"] = 0.0
df["battery_soc"] = 1.0
df = compute_battery_availability(
    df,
    component_name="battery",
    battery_power_column="battery_power",
    battery_soc_column="battery_soc",
    battery_rated_power=BATTERY_RATED_POWER_KW,
    battery_rated_energy=BATTERY_RATED_ENERGY_KWH,
    battery_min_soc=BATTERY_MIN_SOC,
    eta_discharge=ETA_DISCHARGE,
)

mc = MisoCapacity(
    component_list=["wind_farm.power", "battery_availability"],
    class_list=["wind", "storage"],
    df=df[["time_utc", "wind_farm.power", "battery_availability"]],
    zone=ZONE,
    interconnect_limit=INTERCONNECT_LIMIT_KW,
    priority_order=["battery_availability", "wind_farm.power"],
)

# ── Plot 1: SAC stacked bar ──────────────────────────────────────────────────
mc.plot_sac_stacked_bar(save_path=OUTPUT_DIR / "sac_stacked_bar.png")

# ── Plot 2: Revenue stacked bar ──────────────────────────────────────────────
mc.plot_revenue_stacked_bar(save_path=OUTPUT_DIR / "revenue_stacked_bar.png")

# Optional: render both side-by-side in a single figure as well.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
mc.plot_sac_stacked_bar(ax=axes[0])
mc.plot_revenue_stacked_bar(ax=axes[1])
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "sac_and_revenue_stacked_bars.png")

plt.show()

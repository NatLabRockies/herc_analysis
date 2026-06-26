"""Example 05 — Wind + battery MISO capacity: combined analysis and revenue.

Adds a hypothetical 20 MW / 4-hour battery to the Hercules wind dataset and
runs a combined MisoCapacity analysis.  The battery is modelled as always idle
and fully charged (SOC = 1), representing a dedicated capacity resource that
never discharges for energy arbitrage.

Steps:
  1. Compute battery availability with compute_battery_availability.
  2. Build a combined MisoCapacity with wind and battery as two components.
     Battery is given priority over wind in the interconnect allocation so
     that the battery's capacity credits are preserved when the two combined
     would otherwise exceed the interconnect limit.
  3. Print per-component tables and compare revenue with the wind-only case.
"""

from pathlib import Path

from herc_analysis import Scenario
from herc_analysis.capacity import MisoCapacity, compute_battery_availability

DATA_DIR = Path(__file__).parent
ZONE = 1
INTERCONNECT_LIMIT_KW = 61_500.0  # kW

BATTERY_RATED_POWER_KW = 20_000.0  # kW  (20 MW)
BATTERY_RATED_ENERGY_KWH = 80_000.0  # kWh (80 MWh — 4-hour duration)
BATTERY_MIN_SOC = 0.0
ETA_DISCHARGE = 1.0

oa = Scenario(DATA_DIR / "hercules_output.h5")
df = oa.output.df[["time_utc", "wind_farm.power"]].copy()

# ── Wind-only baseline ────────────────────────────────────────────────────────
mc_wind = MisoCapacity(
    component_list=["wind_farm.power"],
    class_list=["wind"],
    df=df[["time_utc", "wind_farm.power"]],
    zone=ZONE,
    interconnect_limit=INTERCONNECT_LIMIT_KW,
    verbose=True,
)

# ── Add battery column via compute_battery_availability ──────────────────────
# Battery is always idle and fully charged — SOC = 1 at every time-step.
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
# compute_battery_availability adds a "battery_availability" column [kW].

# ── Combined wind + battery analysis ─────────────────────────────────────────
# Wind + battery (both at full output) = 61.5 + 20 = 81.5 MW, which exceeds
# the 61.5 MW interconnect limit.  priority_order ensures the battery is fully
# allocated first, with wind absorbing any remaining headroom.
mc_combined = MisoCapacity(
    component_list=["wind_farm.power", "battery_availability"],
    class_list=["wind", "storage"],
    df=df[["time_utc", "wind_farm.power", "battery_availability"]],
    zone=ZONE,
    interconnect_limit=INTERCONNECT_LIMIT_KW,
    priority_order=["battery_availability", "wind_farm.power"],
)

# ── Results ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("WIND-ONLY ANALYSIS")
print("=" * 60)
mc_wind.print_all_component_tables()
wind_only_revenue = mc_wind.get_total_revenue()
print(f"\nWind-only annual revenue: ${wind_only_revenue:,.2f}")

print("\n")
print("=" * 60)
print("COMBINED WIND + BATTERY ANALYSIS")
print("=" * 60)
mc_combined.print_all_component_tables()
combined_revenue = mc_combined.get_total_revenue()
wind_revenue = mc_combined.get_component_revenue("wind_farm.power")
battery_revenue = mc_combined.get_component_revenue("battery_availability")

print(f"\nCombined annual revenue:    ${combined_revenue:,.2f}")
print(f"  Wind contribution:        ${wind_revenue:,.2f}")
print(f"  Battery contribution:     ${battery_revenue:,.2f}")
print(f"\nIncremental battery value:  ${battery_revenue:,.2f}")
print(f"  (combined minus wind-only: ${combined_revenue - wind_only_revenue:,.2f})")

"""Example 03 — Wind-only MISO capacity: tables and revenue summary.

Runs the full MisoCapacity analysis for the wind farm and prints the complete
per-season capacity table alongside the annual revenue breakdown.

Table rows (per season):
  # Hours          — classified hours in the merged RA-hour dataset
  Tier 1 (MW)      — mean wind output during non-RA hours
  Tier 2 (MW)      — mean wind output during RA hours (padded to ≥65 hrs)
  ISAC (MW)        — 0.2·Tier1 + 0.8·Tier2
  Class UCAP (MW)  — MISO reference UCAP for the wind resource class
  Class ISAC (MW)  — MISO reference ISAC for the wind resource class
  SAC (MW)         — Seasonal Accredited Capacity = ISAC × (UCAP/ISAC ratio)
  ZRC (MW)         — Zonal Resource Credits (currently equal to SAC)
  PRA Price ($/MW-day) — clearing price from the PRA auction
  Days             — days in the season used for revenue calculation
  Revenue ($)      — ZRC × PRA Price × Days
"""

from pathlib import Path

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
)

mc.print_all_component_tables()
mc.print_annual_revenue()

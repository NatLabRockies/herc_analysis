# Example analysis of computing MISO capacity from a HERCULES output simulation

`hercules_output.h5` is a HERCULES output simulation of a wind farm in North
Dakota (MISO Zone 1) covering roughly Aug 2023 → Sep 2024 — fully spanning
MISO planning year 2324 (Sept 1, 2023 → Aug 31, 2024) plus partial fringes
of PY 2223 and PY 2425.

## Scripts

- `01_demo_interconnect_limit.py` — effect of an interconnect limit on
  hourly wind output.
- `02_demo_planning_year_coverage.py` — visualize which planning years are
  fully covered by the simulation.
- `03_wind_isac_plots.py` — seasonal Tier 1 / Tier 2 / ISAC bar chart and
  per-season scatter for the wind farm.
- `04_wind_revenue.py` — wind revenue from PRA prices.
- `05_battery_availability.py` — battery availability with the bundled h5.
- `06_wind_battery_analysis.py` — combined wind + battery accreditation.
- `07_sac_revenue_stacked_bars.py` — SAC and revenue stacked-bar charts.

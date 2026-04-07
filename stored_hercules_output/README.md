# Stored Hercules Output

This directory contains output files from real HERCULES simulations, committed
to the repository so that the `examples/` can be run without re-executing
HERCULES.

## Data Sources

### 02b -- Wind Farm Realistic Inflow (Precomputed FLORIS)

Based on [HERCULES Example 02b](https://natlabrockies.github.io/hercules/examples/02b_wind_farm_realistic_inflow_precom_floris.html).

- 9-turbine wind farm, 45 MW interconnect
- 4-hour simulation (Jun 24 2024, 16:59--20:59 UTC)
- Rich per-turbine signals: power, wind speeds, power setpoints

### 05 -- Wind and Storage with LMP

Based on [HERCULES Example 05](https://natlabrockies.github.io/hercules/examples/05_wind_and_storage_with_lmp.html).

- 9-turbine wind farm + 10 MW / 10 MWh battery
- LMP-based battery control (charges when price is low, discharges when high)
- 4-hour simulation (same time window as 02b)
- Logged signals: wind power, battery power/SOC/setpoint, real-time and
  day-ahead LMP

## Notes

- The outputs have been slightly modified from the original HERCULES examples to
  reduce file size.
- The `hercules_input.yaml` and `hercules_runscript.py` files in each directory
  are included for reference only -- they document the exact configuration used
  to produce the output.
- **You do not need to regenerate this data.** It is committed with the
  repository and ready to use.

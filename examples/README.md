# Examples

These examples demonstrate the core analysis workflow provided by `herc_analysis`.
Examples 00, 01, and 05 use pre-computed HERCULES simulation outputs that are
committed with this repository under `stored_hercules_output/`, so there is no
need to run HERCULES first. Examples 02 and 04 additionally need a year-long
`hercules_output.h5` placed in `02_miso_capacity_analysis_wind/` (not
committed; see that folder's `readme.md`).

## 00 -- Single Scenario Analysis

Loads output from HERCULES Example 05 (wind farm + battery storage with
LMP-based control) and shows:

- Component discovery and metrics from a single `Scenario`
  (`scenario.metric_set.to_csv(...)` writes the long-format metrics)
- Interactive plotting with `display.timeseries_figure`, including custom signal
  subplots for wind speed, individual turbine powers, battery SOC, and
  battery power vs setpoint

```bash
cd examples/00_one_scenario_analysis
python example_analysis.py
```

## 01 -- Two Scenario Comparison

Compares two stored scenarios side-by-side:

- **Wind Only** -- HERCULES Example 02b (wind farm with precomputed FLORIS)
- **Wind + Storage** -- HERCULES Example 05 (wind farm + battery + LMP)

Demonstrates cross-scenario comparison with the single `Comparison` engine,
bar-chart plotting of metrics, and multi-scenario interactive overlays.

```bash
cd examples/01_two_scenario_analysis
python example_analysis.py
```

## 02 -- MISO Capacity Analysis

A wind farm spanning a full MISO planning year; demonstrates the `capacity`
package (`MisoCapacity`, availability providers, interconnect limiting, ISAC /
SAC / revenue). See that folder's `readme.md` for the per-script breakdown.

## 04 -- Input Signal Analysis

Exploratory plots of input signals (LMP donuts/histograms/boxplots, diurnal
profiles, correlations) via the `display` input-plot helpers.

## 05 -- Cross-Scenario Comparison

Uses the single `Comparison` engine to compare metrics across an arbitrary
number of cases from tidy per-case `metrics.csv` files: select metrics and an
entity, choose the `annual` or `total` resolution, and render to CSV, a
`great_tables` table, or a per-metric plot.

```bash
cd examples/05_scenario_comparison
python example_scenario_compare.py
```

# Examples

These examples demonstrate the core analysis workflow provided by `herc_analysis`.
They use pre-computed HERCULES simulation outputs that are committed with this
repository under `stored_hercules_output/`, so there is no need to run HERCULES
first.

## 00 -- Single Scenario Analysis

Loads output from HERCULES Example 05 (wind farm + battery storage with
LMP-based control) and shows:

- Component discovery via `OutputAnalysis`
- Total metrics computation with `TotalMetrics`
- Interactive plotting with `PlotHerculesOutput`, including custom signal
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

Demonstrates multi-scenario `TotalMetrics` comparison, bar-chart plotting of
metrics, and multi-scenario interactive overlays.

```bash
cd examples/01_two_scenario_analysis
python example_analysis.py
```

## 05 -- Cross-Scenario Comparison

Uses `ScenarioComparison` to compare metrics across an arbitrary number of
cases from tidy per-case `metrics.csv` files: select metrics and a scope,
choose `annual` or `total` scaling, and render to CSV, a `great_tables` table,
or a per-metric plot.

```bash
cd examples/05_scenario_comparison
python example_scenario_compare.py
```

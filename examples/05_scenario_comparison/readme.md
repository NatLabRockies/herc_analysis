# Example: Cross-Scenario Comparison

Demonstrates `Comparison`, the single engine for comparing metrics across many
simulation cases.

The two bundled stored simulations are treated as two cases:

- **wind_only** -- HERCULES Example 02b (wind farm with precomputed FLORIS)
- **wind_storage** -- HERCULES Example 05 (wind farm + battery + LMP)

`example_scenario_compare.py` shows the full workflow:

1. Build a `Scenario` per case and write its long-format `outputs/metrics.csv`
   via `scenario.metric_set.to_csv(...)` (schema:
   `entity, metric, resolution, period, value, unit, scaling, sim_years`).
2. Collect the per-case files with `Comparison.from_cases` (or compare live
   `Scenario` objects with `Comparison.from_scenarios`).
3. Build comparison tables, selecting metrics, an `entity` (`plant`, a component
   name, or `all`) and a `resolution` (`annual` — average per year, or `total` —
   over the whole simulation length). Both resolutions are materialized up front,
   so the table just reads the requested rows.
4. Render a `great_tables` table and a per-metric plot across cases.

```bash
cd examples/05_scenario_comparison
python example_scenario_compare.py
```

In a real project the per-case `metrics.csv` is produced by a headless analysis
script. `Comparison` can run that script in each case for you:

```python
cmp = Comparison.from_cases(
    cases,
    run_script="compute_metrics.py",  # copied into each case and run headless
)
```

`great-tables` is a core dependency, so `to_great_table(...)` works out of the
box (no extra install).

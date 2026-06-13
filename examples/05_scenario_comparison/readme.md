# Example: Cross-Scenario Comparison

Demonstrates `ScenarioComparison`, the generic engine for comparing metrics
across many simulation cases.

The two bundled stored simulations are treated as two cases:

- **wind_only** -- HERCULES Example 02b (wind farm with precomputed FLORIS)
- **wind_storage** -- HERCULES Example 05 (wind farm + battery + LMP)

`example_scenario_compare.py` shows the full workflow:

1. Write a tidy per-case `outputs/metrics.csv` (the schema the engine reads):
   `scope, metric, value, unit, scaling, sim_years`.
2. Collect the per-case files with `ScenarioComparison.from_cases`.
3. Build `annual` and `total` comparison tables, selecting metrics and a scope
   (`plant`, a component name, or `all`). Scaling is handled automatically via
   each metric's `scaling` tag (`extensive`, `intensive`, or `annual`).
4. Render a `great_tables` table (optional dependency) and a per-metric plot
   across cases.

```bash
cd examples/05_scenario_comparison
python example_scenario_compare.py
```

In a real project the tidy `metrics.csv` is produced by a headless analysis
script. `ScenarioComparison` can run that script in each case for you:

```python
sc = ScenarioComparison.from_cases(
    cases,
    run_script="compute_metrics.py",  # copied into each case and run headless
)
```

The `great_tables` rendering requires the optional extra:

```bash
pip install great-tables   # or: pip install herc_analysis[tables]
```

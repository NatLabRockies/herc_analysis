# Migrating to the rebuilt herc_analysis

This guide moves a downstream project (a per-study `analysis.py` / `analyze.py`
that imports `herc_analysis`) from the pre-refactor API onto the rebuilt,
layered API. The numbers do not change — the refactor reproduced every existing
capability — only the import paths and a few object names do.

## TL;DR

- One `Scenario` replaces the `OutputAnalysis` + `TotalMetrics` pair.
- Metrics are one **long-format** table (`MetricSet`) keyed by
  `(entity, metric, resolution, period)`, not a nested dict. `Scenario.metrics`
  still returns the familiar nested dict; `Scenario.metric_set` is the long form.
- There is **one** cross-scenario engine, `Comparison`
  (`TotalMetrics([...])` and `ScenarioComparison` are gone).
- Capacity lives in `herc_analysis.capacity`; plotting/tables/input plots live in
  `herc_analysis.display`.
- The pre-refactor classes were **removed** (no overlap window): update imports
  in one pass using the table below.

## Install / version bump

1. Bump the `herc_analysis` pin in your project's `pyproject.toml` to the
   refactored release.
2. **Before** touching your code, capture a baseline from the *old* version:
   run your existing analysis once and save its metrics output (e.g.
   `outputs/metrics.csv` or the pickled nested dict). You will diff against this
   to prove the port is numerically faithful.
3. `uv sync` (or `pip install -e .`) the new version.

## Concepts to absorb first

1. **One object per run.** `Scenario(path)` discovers components cheaply;
   `scenario.channels` (derived time series) and `scenario.metrics` /
   `scenario.metric_set` compute lazily and cache. The raw frame is not copied —
   it stays on `scenario.output` (a `HerculesOutput`).
2. **Metrics are long-format.** `scenario.metric_set` is a table with one row per
   `(entity, metric, resolution, period)` plus `value`, `unit`, `scaling`. Use
   `.scalar(entity, metric)` to read one value and `.to_csv(path)` to export. For
   the *exact* pre-refactor nested dict use `scenario.metrics`;
   `MetricSet.to_nested()` is a lighter `{entity: {metric: value}}` grouping (no
   `simulation_metadata` / `component_type` / `categories` nesting).
3. **Units are explicit in channel names.** Derived channels use
   `{component}__{signal}`: power stays in **kW** (`battery__power_kw`), energy is
   **MWh** (`battery__energy_mwh`), revenue is **$** (`battery__revenue_rt_usd`),
   and plant aggregates use a `plant__` prefix. (Metric *names* in `metric_set`
   keep the friendly `energy_mwh` / `revenue_rt` form.)

## Symbol-by-symbol map

| Pre-refactor | Now |
|---|---|
| `from herc_analysis import OutputAnalysis` | `from herc_analysis import Scenario` |
| `OutputAnalysis(file)` | `Scenario(file)` |
| `oa.df` | `scenario.channels` (new names) / `scenario.output.df` (raw signals) |
| `oa.interconnect_mw`, `oa.dt`, `oa.total_simulation_time_s` | `scenario.meta.interconnect_mw`, `scenario.meta.dt_s`, `scenario.meta.total_simulation_time_s` |
| `oa.components`, `oa.generators`, `oa.storage` | same on `scenario` |
| `TotalMetrics(oa).compute_metrics()` | `scenario.metrics` (nested dict) |
| `TotalMetrics(oa).compute_monthly_metrics()` | `scenario.monthly_metrics` |
| `tm.save_metrics("metrics.csv")` | `scenario.metric_set.to_csv("metrics.csv")` |
| `TotalMetrics([oa1, oa2])` + `compare_scenarios()` | `Comparison.from_scenarios([s1, s2]).table(...)` |
| `ScenarioComparison.from_cases(dirs)` | `Comparison.from_cases(dirs)` |
| `sc.compare(metrics, period=, scope=)` | `cmp.table(metrics, view=, entity=)` (`scope`→`entity`; `period="annual"/"total"`→`view="per_year"/"cumulative"`) |
| `sc.to_great_table(df)` / `sc.plot_metric(...)` | `cmp.to_great_table(df)` / `cmp.plot(...)` |
| `from herc_analysis.miso_capacity import MisoCapacity` | `from herc_analysis.capacity import MisoCapacity` |
| `compute_battery_availability(...)` (free fn) | `from herc_analysis.capacity import BatteryAvailability` (provider) — the free function is still importable from `herc_analysis.capacity` |
| `PlotHerculesOutput(oa)` | `from herc_analysis.display import timeseries_figure` → `timeseries_figure(scenario)` |
| `from herc_analysis.input_analysis import plot_*` | `from herc_analysis.display import plot_*` (also re-exported at `herc_analysis` top level) |
| `from herc_analysis.utilities import interpolate_df, add_local_time` | `from herc_analysis.timeseries import interpolate_df, add_local_time` |

Argument-level notes:

- `Comparison.table(..., entity=...)` replaces `scope=...`; pass `entity="all"`
  for a `(entity, metric)` MultiIndex, or a list of entities.
- `Comparison.table`/`plot` take `view="per_year"` (default) or
  `view="cumulative"` — the rename of the old `period="annual"/"total"`.
- `Comparison.table(..., resolution=...)` selects which resolution's rows to
  compare (`"total"` by default).
- `Scenario(..., name=..., resolutions=("total", "annual"))` are new keyword args.
  Resolutions: `total` (whole-run total), `annual` (the *average* annual value =
  extensive totals ÷ `sim_years`), `yearly` (per specific calendar year),
  `monthly`. `yearly`/`monthly` are opt-in. Read with
  `scenario.metric_set.scalar(entity, metric, resolution=, period=)` or
  `scenario.metric_set.at(resolution)`.

## Worked example A — one case (`analyze.py`)

```python
# ---------- BEFORE ----------
from herc_analysis import OutputAnalysis, TotalMetrics, PlotHerculesOutput
from herc_analysis.miso_capacity import MisoCapacity, compute_battery_availability

oa = OutputAnalysis("outputs/hercules_output.h5")
tm = TotalMetrics(oa)
metrics = tm.compute_metrics()                 # nested dict
tm.save_metrics("outputs/metrics.csv")

avail = compute_battery_availability(oa.df, ...)
cap = MisoCapacity(["battery"], ["storage"], oa.df[["time_utc", "battery"]],
                   zone=4, interconnect_limit=..., pra_years=[2024])
fig = PlotHerculesOutput(oa); fig.plot_interactive(save_file="outputs/timeseries.html")

# ---------- AFTER ----------
from herc_analysis import Scenario
from herc_analysis.capacity import MisoCapacity, BatteryAvailability
from herc_analysis.display import timeseries_figure

s = Scenario("outputs/hercules_output.h5", name="wind_storage")
metrics = s.metrics                            # same nested dict
s.metric_set.to_csv("outputs/metrics.csv")     # long-format

cap = MisoCapacity.from_scenario(
    s, zone=4, pra_years=[2024],
    availability={"battery": BatteryAvailability(
        rated_power_kw=..., rated_energy_kwh=..., min_soc=..., eta_discharge=...,
    )},
)
cap.report()

timeseries_figure(s).plot_interactive(save_file="outputs/timeseries.html")
```

## Worked example B — many cases (`analysis.py`)

```python
# ---------- BEFORE ----------
from herc_analysis import OutputAnalysis, TotalMetrics
from herc_analysis.scenario_compare import ScenarioComparison

oas = [OutputAnalysis(p) for p in case_paths]
tm = TotalMetrics(oas)                                            # list-mode
tm.compute_metrics(); tm.compare_scenarios(output_csv="outputs/compare.csv")
sc = ScenarioComparison.from_cases(case_dirs, metric_file="outputs/metrics.csv")
sc.to_great_table(sc.table(["capacity_factor", "revenue_rt"]))

# ---------- AFTER ----------
from herc_analysis import Comparison, Scenario

def analyze_one(path, name):
    s = Scenario(path, name=name)
    s.metric_set.to_csv(f"outputs/{name}/metrics.csv")
    return s

scenarios = [analyze_one(p, n) for p, n in zip(case_paths, case_names)]

cmp = Comparison.from_scenarios(scenarios)                        # one engine...
# ...or from saved files (the case-directory workflow, unchanged):
cmp = Comparison.from_cases(case_dirs, metric_file="outputs/metrics.csv")

cmp.to_great_table(cmp.table(["capacity_factor", "revenue_rt"], view="per_year"))
cmp.plot("revenue_rt", entity="plant", view="per_year")
```

## Rename / shim notes

- **Channel columns.** A notebook that read `oa.df["battery_power_mw"]` should
  read `scenario.channels["battery__power_kw"] / 1000` (or just
  `scenario.channels["battery__power_kw"]` in kW). Raw Hercules signals
  (`wind_farm.wind_speed_mean_background`, `battery.soc`, …) live on
  `scenario.output.df`. The bundled `timeseries_figure` rebuilds the old-style
  names internally, so existing `SignalSubplot` specs keep working — see
  [`PLOTTING.md`](PLOTTING.md) for a `plot_interactive` / `SignalSubplot`
  cheat sheet.
- **Nested dict.** Code that wants the exact pre-refactor nested dict uses
  `scenario.metrics`; `scenario.metric_set.to_nested()` gives a lighter
  `{entity: {metric: value}}` view.

## Verification recipe

1. Re-run your ported script and write the new `outputs/metrics.csv`.
2. Diff the metric values against the baseline you captured on the old version.
   Because Hercules stores `float32`, agreement is at float32 precision
   (`rtol≈1e-6`); differences below that come from summation order, not behavior.
3. Spot-check the trusted anchors, e.g. plant capacity factor and total energy.
4. Regenerate any figures and eyeball them against the originals.

## Troubleshooting

- **`AttributeError: 'Scenario' object has no attribute 'df'`** — use
  `scenario.channels` (derived) or `scenario.output.df` (raw).
- **A metric returns `NaN`** — check the `(entity, metric, resolution, period)`
  spelling passed to `MetricSet.scalar` / `Comparison.table`; an unknown
  combination yields `NaN` rather than raising.
- **Capacity revenue is `NaN`** — short simulation windows are dropped by the
  MISO low-hour planning-year filter; pass
  `remove_low_hour_planning_years=False` for short test windows.
- **Unit surprise** — channels store power in **kW**; a value that looks 1000×
  off is a kW-vs-MW mix. The single conversion lives in `channels.energy_mwh` /
  `channels.kw_to_mw`.
- **Mixed old/new metric files** — `Comparison` expects the `entity` schema; a
  legacy file using `scope` must be renamed `scope` → `entity` first.

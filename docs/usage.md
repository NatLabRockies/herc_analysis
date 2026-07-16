# Usage

herc_analysis is organized as layers that flow strictly upward — a lower layer
never imports a higher one:

| Layer | Module | Role |
|-------|--------|------|
| L0 io | `herc_analysis.io` | Thin boundary over `HerculesOutput` (`RunMeta`, `load_run`). |
| L1 pure | `channels`, `pricing`, `reducers`, `timeseries`, `components` | Vectorized free functions; the bulk of the tested logic. |
| L2 core | `herc_analysis.scenario` | `Scenario` — one run's channels + metrics. |
| L3 metrics | `herc_analysis.metrics` | `MetricSet` — the long-format metrics table. |
| L4 capacity | `herc_analysis.capacity` | `CapacityBase` → `MisoCapacity`; availability providers. |
| L5 agg | `herc_analysis.comparison` | `Comparison` — the single cross-scenario engine. |
| L6 display | `herc_analysis.display` | Plotly figure, input plots, `great_tables` rendering. |

## Single scenario

```python
from herc_analysis import Scenario

s = Scenario("outputs/hercules_output.h5", name="wind_storage")
s.metrics                       # nested-dict metrics for the whole run
s.metric_set.scalar("plant", "capacity_factor")
s.metric_set.to_csv("outputs/metrics.csv")   # long-format (entity, metric, ...)
```

`Scenario` construction is cheap; `channels` and the metrics compute on first
access and are cached. The raw frame is never copied — it stays on `s.output`.

## Comparing scenarios

```python
from herc_analysis import Comparison

cmp = Comparison.from_scenarios([s1, s2])          # or .from_cases(case_dirs)
cmp.table(["energy_mwh", "revenue_rt"], resolution="annual", entity="plant")
cmp.plot("energy_mwh", resolution="total")
```

`resolution` selects which view of each metric to compare: `"total"` is the
value over the whole simulation length, `"annual"` is the average value per year.
Both are materialized up front, so the table just reads the requested rows — no
rescaling at query time. `resolution` defaults to `"total"` (same as
`MetricSet.scalar`).

`Scenario` exposes metrics at several **resolutions** (choose with
`Scenario(file, resolutions=...)`, default `("total", "annual")`):

| resolution | meaning |
|-----------|---------|
| `total`   | value over the whole simulation length |
| `annual`  | average annual value (extensive totals ÷ `sim_years`) |
| `yearly`  | per specific calendar year (`"2024"`, `"2025"`, …) |
| `monthly` | per calendar month (`"2024-03"`, …) |

Read one with `scenario.metric_set.scalar(entity, metric, resolution=..., period=...)`
or a slice with `scenario.metric_set.at("yearly")`. `Comparison.table(..., resolution=...)`
selects which resolution's rows to compare.

## Capacity accreditation

```python
from herc_analysis.capacity import MisoCapacity, BatteryAvailability

cap = MisoCapacity.from_scenario(
    s, zone=4, pra_years=[2024],
    availability={"battery": BatteryAvailability(
        rated_power_kw=..., rated_energy_kwh=..., min_soc=..., eta_discharge=...,
    )},
)
cap.revenue()        # annual $ per component
cap.to_metrics()     # long-format rows at annual + total resolution for Comparison
```

## Display

```python
from herc_analysis.display import timeseries_figure

timeseries_figure(s).plot_interactive(save_file="outputs/timeseries.html")
```

See `PLOTTING.md` (repo root) for a `plot_interactive` / `SignalSubplot` cheat
sheet with copy-paste recipes.

Migrating an existing project from the pre-refactor API? See `MIGRATION.md` in
the repository root.

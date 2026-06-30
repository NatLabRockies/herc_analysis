# herc_analysis

Analysis tools for Hercules hybrid energy system simulations.

Hercules: https://github.com/NatLabRockies/hercules

## Clone the repository

```bash
git clone https://github.com/NatLabRockies/herc_analysis
cd herc_analysis
```

## Installation

Typical installation is to install into same environment as Hercules is installed in.

### UV Based Installation (Basic):

Can use `uv sync` to install all dependencies and Hercules:

```bash
uv sync
```

or create a new environment and install:

```bash
uv venv
uv pip install -e .
```

### UV Based Installation (Developer):

```bash
uv sync --all-extras
uv run pre-commit install
```

or create a new environment and install:

```bash
uv venv
uv pip install -e .[docs,develop]
```


### Pip Based Installation:
```bash
pip install -e .
```

### Pip Based Installation (Developer):
```bash
pip install -e .[docs,develop]
pre-commit install
```

## Quickstart

The package is built from atomic building blocks that compose upward: a thin
boundary over Hercules, a library of pure vectorized functions, and a few
orchestrating objects on top.

```python
from herc_analysis import Scenario, Comparison
from herc_analysis.capacity import MisoCapacity, BatteryAvailability
from herc_analysis.display import timeseries_figure

# One Hercules run -> channels + metrics (lazy, cached).
s = Scenario("outputs/hercules_output.h5", name="wind_storage")
s.metric_set.to_csv("outputs/metrics.csv")        # long-format metrics

# Capacity accreditation (MISO today; SPP slots in beside it).
cap = MisoCapacity.from_scenario(s, zone=4, pra_years=[2024])
cap.report()                                       # seasonal + annual tables

# Compare many runs through one engine.
cmp = Comparison.from_scenarios([s, other_scenario])
cmp.table(["energy_mwh", "capacity_factor", "revenue_rt"], view="per_year")

# Interactive time-series figure.
timeseries_figure(s).plot_interactive(save_file="outputs/timeseries.html")
```

Key objects:

* `Scenario` — analysis of one run (`.channels`, `.metrics`, `.metric_set`).
* `MetricSet` — the long-format metrics table shared by `Scenario` and `Comparison`.
* `Comparison` — the single cross-scenario engine (`from_scenarios` / `from_cases`).
* `herc_analysis.capacity` — `MisoCapacity` + availability providers.
* `herc_analysis.display` — `timeseries_figure`, input plots, `to_great_table`.

Moving from a pre-refactor project? See [`MIGRATION.md`](MIGRATION.md).

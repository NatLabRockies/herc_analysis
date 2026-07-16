# Plotting cheat sheet

The interactive time-series figure is built from a `Scenario` (or a list of them)
via `display.timeseries_figure`, then rendered with `.plot_interactive(...)`.
Custom panels are described with `SignalSubplot` specs.

```python
from herc_analysis import Scenario, SignalSubplot
from herc_analysis.display import timeseries_figure

s = Scenario("outputs/hercules_output.h5", name="wind_storage")
plotter = timeseries_figure(s)

fig = plotter.plot_interactive(save_file="outputs/timeseries.html")
```

`plot_interactive` returns a Plotly `Figure` (so you can keep tweaking it) and,
when `save_file` is given, writes an HTML file.

---

## `plot_interactive` options

| Argument | Default | What it does |
|---|---|---|
| `plot_dt` | `60` | Downsample interval in **seconds** before plotting (bigger = fewer points = faster). |
| `height_per_subplot` | `300` | Pixel height of each stacked subplot. |
| `save_file` | `None` | HTML output path. `None` = don't write a file. |
| `open_browser` | `True` | Open the saved file in a browser. Set `False` for headless/CI. |
| `show_plant_power` | `True` | Include the stacked plant-power subplot. |
| `show_market` | `True` | Include the market-price subplot (always the bottom row when on). |
| `show_negative_ptc_line` | `True` | Draw the `-PTC` reference line on the market subplot. |
| `shade_price_area` | `True` | Shade positive/negative price regions. |
| `show_interconnect_limit` | `True` | Draw the interconnect-limit line on plant power. |
| `date_range` | `None` | Zoom to `["YYYY-MM-DD", "YYYY-MM-DD"]`. |
| `signal_subplots` | `None` | List of `SignalSubplot` panels stacked above plant power / market. |
| `da_only` | `False` | Market subplot shows only the day-ahead price (RT hidden), filled to zero. |

Panel order (top → bottom): custom `signal_subplots`, then plant power, then
market.

---

## `SignalSubplot` fields

```python
SignalSubplot(
    columns,                      # str or list[str] — column name(s) to plot
    title="",                     # panel title (auto from columns if "")
    y_label="",                   # y-axis label
    scale=1.0,                    # multiply every y value (e.g. 100 for 0-1 -> %)
    reference_lines=None,         # list[float] — horizontal lines to draw
    reference_line_labels=None,   # list[str] — labels (must match reference_lines len)
)
```

### What column names can I use?

List them for your run:

```python
plotter.print_available_signals()
```

Two families are available on a `Scenario`-backed figure:

- **Derived (MW / friendly names):** `total_plant_power_mw`, `lmp_rt`, `lmp_da`,
  `{name}_power_mw`, `{name}_power_setpoint_mw`, `{name}_soc` (storage), …
- **Raw Hercules signals (dotted):** `battery.soc`, `wind_farm.wind_speed_mean_background`,
  `wind_farm.wind_speed_mean_withwakes`, `wind_farm.turbine_powers.000`,
  `solar_farm.poa`, … (anything in `s.output.df`).

> Note: `Scenario.channels` stores power in **kW** (`battery__power_kw`), but the
> plotter exposes the old-style **MW** name `battery_power_mw` for convenience —
> use the names `print_available_signals()` reports.

---

## Copy-paste recipes

### Battery SOC, scaled to % with min/max guide lines

```python
SignalSubplot(
    columns="battery_soc",
    title="Battery State of Charge",
    y_label="SOC (%)",
    scale=100,                       # 0-1 fraction -> percent
    reference_lines=[10, 90],
    reference_line_labels=["Min SOC", "Max SOC"],
)
```

### Several signals on one panel (e.g. wind speeds)

```python
SignalSubplot(
    columns=[
        "wind_farm.wind_speed_mean_background",
        "wind_farm.wind_speed_mean_withwakes",
    ],
    title="Wind Speed",
    y_label="Wind Speed (m/s)",
)
```

### A component's power (+ setpoint) — helper

```python
plotter.component_power_subplot("battery")     # returns a SignalSubplot
```

### First few turbine powers

```python
turbine_cols = sorted(
    c for c in s.output.df.columns if c.startswith("wind_farm.turbine_powers.")
)
SignalSubplot(
    columns=turbine_cols[:3],
    title="Turbine Powers (first 3)",
    y_label="Power (kW)",
)
```

### Put it together — a full figure

```python
plotter.plot_interactive(
    save_file="outputs/timeseries_full.html",
    plot_dt=10,                      # 10-second resolution
    signal_subplots=[
        SignalSubplot("wind_farm.wind_speed_mean_background",
                      title="Wind Speed", y_label="m/s"),
        plotter.component_power_subplot("battery"),
        SignalSubplot("battery_soc", title="Battery SOC", y_label="SOC (%)",
                      scale=100, reference_lines=[10, 90]),
    ],
)
```

### Market panel only (no plant power, no custom signals)

```python
plotter.plot_interactive(show_plant_power=False, signal_subplots=None)
```

### Day-ahead price only, shaded

```python
plotter.plot_interactive(da_only=True, shade_price_area=True)
```

### Zoom to a date window

```python
plotter.plot_interactive(date_range=["2024-03-01", "2024-03-08"])
```

### Headless / scripted (no browser, just save)

```python
plotter.plot_interactive(save_file="outputs/ts.html", open_browser=False)
```

### Overlay multiple scenarios

```python
plotter = timeseries_figure([s1, s2], scenario_names=["Wind Only", "Wind + Storage"])
plotter.plot_interactive(
    signal_subplots=[plotter.component_power_subplot("wind_farm")],
)
```

---

See `examples/00_one_scenario_analysis/` and
`examples/01_two_scenario_analysis/` for runnable versions.

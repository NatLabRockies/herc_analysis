"""L6 -- the signal-driven Plotly time-series figure.

The signal-driven plotter engine lives in :mod:`._plot_engine` (relocated from
the former top-level ``plotting`` module). ``timeseries_figure`` is the entry
point: a small factory that accepts a ``Scenario`` (or a list, for the
multi-scenario overlay mode) and adapts it to the interface the plotter expects
-- ``.df`` (derived + raw signal columns), ``.components`` and
``.interconnect_mw``. Any other object already exposing that interface is passed
through unchanged. The plotter itself is unmodified, so figures match the
pre-refactor output.
"""

from __future__ import annotations

from functools import cached_property

import pandas as pd

from herc_analysis.display._plot_engine import (
    PlotHerculesOutput as PlotHerculesOutput,
)


def _legacy_plot_df(scenario) -> pd.DataFrame:
    """Rebuild an ``OutputAnalysis``-style frame from a ``Scenario`` for plotting.

    Maps the new channel convention back to the names the signal-driven plotter
    expects (``{name}_power_mw``, ``total_plant_power_mw``, ``{name}_soc``, ...)
    and merges in the raw Hercules signal columns (those containing a dot) so
    existing ``SignalSubplot`` specs keep working.
    """
    ch = scenario.channels
    df = pd.DataFrame(index=ch.index)

    for col in ("time", "time_utc", "lmp_rt", "lmp_da", "lmp_rt_hourly"):
        if col in ch.columns:
            df[col] = ch[col]
    if "plant__total_power_mw" in ch.columns:
        df["total_plant_power_mw"] = ch["plant__total_power_mw"]

    for comp in scenario.components:
        n = comp.name
        renames = {
            f"{n}__power_kw": (f"{n}_power_mw", 1 / 1000.0),
            f"{n}__power_setpoint_kw": (f"{n}_power_setpoint_mw", 1 / 1000.0),
            f"{n}__energy_mwh": (f"{n}_energy_mwh", 1.0),
            f"{n}__revenue_rt_usd": (f"{n}_revenue_rt", 1.0),
            f"{n}__revenue_da_usd": (f"{n}_revenue_da", 1.0),
            f"{n}__soc": (f"{n}_soc", 1.0),
        }
        for src, (dst, factor) in renames.items():
            if src in ch.columns:
                df[dst] = ch[src] * factor if factor != 1.0 else ch[src]

    # Merge raw Hercules signal columns (e.g. wind_farm.wind_speed_mean_*,
    # battery.soc), deduped on time the same way Scenario.channels was built.
    raw = scenario.output.df.copy()
    raw["time"] = raw["time"].astype(float)
    if raw["time"].nunique() != len(raw):
        raw = raw.drop_duplicates(subset=["time"], keep="first")
    raw = raw.reset_index(drop=True)
    for col in raw.columns:
        if "." in col and col not in df.columns:
            df[col] = raw[col]

    return df


class _ScenarioPlotAdapter:
    """Adapt a ``Scenario`` to the ``.df`` / ``.components`` / ``.interconnect_mw``
    interface ``PlotHerculesOutput`` consumes."""

    def __init__(self, scenario):
        self._scenario = scenario
        self.components = scenario.components
        self.interconnect_mw = scenario.meta.interconnect_mw

    @cached_property
    def df(self) -> pd.DataFrame:
        return _legacy_plot_df(self._scenario)

    def _get_time_axis_info(self):
        """Mirror ``OutputAnalysis._get_time_axis_info`` for the plotter."""
        if "time_utc" in self.df.columns and self.df["time_utc"].notna().any():
            return "time_utc", "Time (UTC)"
        return "time", "Time (seconds)"


def _adapt(source):
    """Wrap a Scenario in the plot adapter; pass other objects through."""
    if hasattr(source, "channels") and hasattr(source, "meta"):
        return _ScenarioPlotAdapter(source)
    return source


def timeseries_figure(source, scenario_names: list[str] | None = None):
    """Build the interactive time-series plotter for a scenario (or list).

    Args:
        source (Scenario | list): A single ``Scenario`` (or any object already
            exposing the ``.df`` / ``.components`` / ``.interconnect_mw``
            plot interface), or a list for the multi-scenario overlay mode.
        scenario_names (list[str], optional): Labels for multi-scenario plots.
            Defaults to None.

    Returns:
        PlotHerculesOutput: The plotter; call ``.plot_interactive(...)`` to
        produce (and optionally save) the figure, or
        ``.print_available_signals()`` to list plottable columns.

    Raises:
        ValueError: If ``source`` is an empty list.
    """
    if isinstance(source, list):
        if not source:
            raise ValueError("source list is empty; pass at least one Scenario.")
        adapted = [_adapt(s) for s in source]
    else:
        adapted = _adapt(source)
    return PlotHerculesOutput(adapted, scenario_names=scenario_names)

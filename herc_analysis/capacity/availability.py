"""L4 -- availability providers.

The recurring need from the notes: a resource's availability for capacity
accounting often comes from somewhere other than the run being analyzed -- a
fixed assumption, a signal pulled from another (unconstrained) run, or computed
from in-run data (the battery case).

Rather than a formal interface, a "provider" is any object with an
``attach(df, name) -> df`` method that returns a copy of ``df`` with a
``{name}__availability`` column added. The three classes below follow that
convention so they stay swappable with nothing to inherit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from herc_analysis.capacity._miso_engine import compute_battery_availability


class FixedAvailability:
    """Constant availability (e.g. 'always available' at a rated value).

    Args:
        value_kw (float): The constant availability value in kW.
    """

    def __init__(self, value_kw: float):
        self.value_kw = value_kw

    def attach(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Return a copy of ``df`` with a constant ``{name}__availability`` column."""
        out = df.copy()
        out[f"{name}__availability"] = self.value_kw
        return out


class FromRunAvailability:
    """Availability lifted from another run/source, aligned by ``time_utc``.

    The generic "signal from another file" case: read a precomputed availability
    signal from a second source and align it onto ``df`` by timestamp.

    Args:
        source (pd.DataFrame | str | Path): A DataFrame, or a path to a CSV /
            parquet file, containing ``time_column`` and ``source_column``.
        source_column (str): Column in ``source`` holding the availability
            signal (kW).
        time_column (str): Timestamp column used to align. Defaults to
            ``"time_utc"``.
    """

    def __init__(
        self,
        source: pd.DataFrame | str | Path,
        source_column: str,
        *,
        time_column: str = "time_utc",
    ):
        self.source = source
        self.source_column = source_column
        self.time_column = time_column

    def _load(self) -> pd.DataFrame:
        if isinstance(self.source, pd.DataFrame):
            return self.source
        path = Path(self.source)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, parse_dates=[self.time_column])

    def attach(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Return a copy of ``df`` with ``{name}__availability`` aligned by time."""
        src = self._load()
        mapping = src.set_index(self.time_column)[self.source_column]
        out = df.copy()
        out[f"{name}__availability"] = out[self.time_column].map(mapping)
        return out


class BatteryAvailability:
    """Compute availability from in-run battery power + SOC.

    Wraps the trusted ``compute_battery_availability`` (the same hourly
    output-potential calculation used by the current capacity pipeline) as an
    injectable provider, renaming its output to the ``{name}__availability``
    channel convention.

    Args:
        rated_power_kw (float): Battery rated power [kW].
        rated_energy_kwh (float): Battery rated energy [kWh].
        min_soc (float): Minimum usable SOC [0-1].
        eta_discharge (float): Discharge efficiency [0-1].
        power_column (str, optional): Battery power column. Defaults to
            ``"{name}__power_kw"`` resolved at attach time.
        soc_column (str, optional): Battery SOC column. Defaults to
            ``"{name}__soc"`` resolved at attach time.
    """

    def __init__(
        self,
        *,
        rated_power_kw: float,
        rated_energy_kwh: float,
        min_soc: float,
        eta_discharge: float,
        power_column: str | None = None,
        soc_column: str | None = None,
    ):
        self.rated_power_kw = rated_power_kw
        self.rated_energy_kwh = rated_energy_kwh
        self.min_soc = min_soc
        self.eta_discharge = eta_discharge
        self.power_column = power_column
        self.soc_column = soc_column

    def attach(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Return a copy of ``df`` with a computed ``{name}__availability`` column."""
        power_column = self.power_column or f"{name}__power_kw"
        soc_column = self.soc_column or f"{name}__soc"
        out = compute_battery_availability(
            df,
            component_name=name,
            battery_power_column=power_column,
            battery_soc_column=soc_column,
            battery_rated_power=self.rated_power_kw,
            battery_rated_energy=self.rated_energy_kwh,
            battery_min_soc=self.min_soc,
            eta_discharge=self.eta_discharge,
        )
        return out.rename(columns={f"{name}_availability": f"{name}__availability"})

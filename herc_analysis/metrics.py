"""L3 -- the long-format metrics table.

``MetricSet`` is the canonical long-format metric output: one row per
``(entity, metric, resolution, period)``. It is the shared link between a single
:class:`~herc_analysis.scenario.Scenario` and the cross-scenario
:class:`~herc_analysis.comparison.Comparison` -- both speak this one shape.

It keeps the scaling semantics designed in the original ``ScenarioComparison``
tidy-file format (``extensive`` / ``intensive`` / ``annual``), renaming that
file's ``scope`` column to ``entity`` and adding the general ``resolution`` /
``period`` axis. ``to_nested()`` reconstructs a nested ``{entity: {metric: value}}``
dict for back-compat consumers.

Scaling semantics (carried over unchanged):

* ``extensive``  -- cumulative over time; the stored value is the simulation
  total. annual = value / sim_years; total = value.
* ``intensive``  -- a rate/ratio; unchanged by simulation length.
  annual = value; total = value.
* ``annual``     -- already per-year (e.g. capacity-auction revenue).
  annual = value; total = value * sim_years.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

Scaling = Literal["extensive", "intensive", "annual"]
Resolution = Literal["total", "annual", "yearly", "monthly"]

METRIC_COLUMNS = (
    "entity",
    "metric",
    "resolution",
    "period",
    "value",
    "unit",
    "scaling",
)


@dataclass(frozen=True)
class MetricSpec:
    """How one metric is named, united, and scaled. A pure record.

    Attributes:
        name (str): Metric name (e.g. ``"energy_mwh"``).
        unit (str): Human-readable unit (e.g. ``"MWh"``, ``"k$"``, ``"-"``).
        scaling (str): One of ``"extensive"``, ``"intensive"``, ``"annual"``.
    """

    name: str
    unit: str
    scaling: Scaling


class MetricSet:
    """A run's metrics as a long-format table: one row per
    ``(entity, metric, resolution, period)``.

    ``entity``     -- ``"plant"``, a category, or a component name.
    ``resolution`` -- ``total`` (whole-run total), ``annual`` (the average
                      annual value), ``yearly`` (per calendar year) or
                      ``monthly``.
    ``period``     -- the bucket within the resolution
                      (``"total"``, ``"annual"``, ``"2024"``, ``"2024-03"``).
    ``scaling``    -- extensive / intensive / annual normalization (the metric's
                      intrinsic nature, the same at every resolution).
    """

    def __init__(self, rows: pd.DataFrame, *, sim_years: float):
        """Wrap a long-format metrics frame.

        Args:
            rows (pd.DataFrame): Long-format rows with at least
                :data:`METRIC_COLUMNS`.
            sim_years (float): Simulation length in years (constant for the run).

        Raises:
            ValueError: If required columns are missing.
        """
        missing = set(METRIC_COLUMNS) - set(rows.columns)
        if missing:
            raise ValueError(f"rows is missing required columns: {sorted(missing)}")
        self.rows = rows[list(METRIC_COLUMNS)].reset_index(drop=True)
        self.sim_years = sim_years

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def scalar(
        self,
        entity: str,
        metric: str,
        *,
        resolution: str = "total",
        period: str = "total",
    ) -> float:
        """Look up a single metric value.

        Args:
            entity (str): Entity name (``"plant"``, a category, or component).
            metric (str): Metric name.
            resolution (str): Temporal resolution. Defaults to ``"total"``.
            period (str): Bucket within the resolution. Defaults to ``"total"``.

        Returns:
            float: The value, or NaN if no matching row exists.
        """
        m = self.rows
        hit = m[
            (m["entity"] == entity)
            & (m["metric"] == metric)
            & (m["resolution"] == resolution)
            & (m["period"] == period)
        ]
        return float(hit["value"].iloc[0]) if len(hit) else float("nan")

    def at(self, resolution: str) -> pd.DataFrame:
        """Return all rows at one resolution (e.g. ``.at("monthly")``)."""
        return self.rows[self.rows["resolution"] == resolution].copy()

    def to_long(self, *, with_sim_years: bool = True) -> pd.DataFrame:
        """Return the long-format rows, optionally with a ``sim_years`` column.

        Args:
            with_sim_years (bool): Append the constant ``sim_years`` column.
                Defaults to True.

        Returns:
            pd.DataFrame: The long-format metrics.
        """
        out = self.rows.copy()
        if with_sim_years:
            out["sim_years"] = self.sim_years
        return out

    def to_csv(self, path: str | Path) -> Path:
        """Write the long-format table (with ``sim_years``) to CSV.

        Args:
            path (str | Path): Output path.

        Returns:
            Path: The path written.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.to_long().to_csv(out, index=False)
        return out

    def to_nested(self, *, resolution: str = "total") -> dict:
        """Rebuild a nested ``{entity: {metric: value}}`` dict for one resolution.

        Back-compat view for callers that prefer nested access. Note this is the
        long table's own entity/metric grouping, not the full
        ``Scenario.metrics`` dict (which keeps extra structure like
        ``simulation_metadata`` and per-component ``component_type``).

        Args:
            resolution (str): Resolution to extract. Defaults to ``"total"``.

        Returns:
            dict: ``{entity: {metric: value}}`` for the chosen resolution
            (period ``"total"`` for the total resolution, else all periods are
            collapsed to the first occurrence).
        """
        sub = self.rows[self.rows["resolution"] == resolution]
        nested: dict = {}
        for _, r in sub.iterrows():
            nested.setdefault(r["entity"], {})[r["metric"]] = float(r["value"])
        return nested

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_long(
        cls, df: pd.DataFrame, *, sim_years: float | None = None
    ) -> MetricSet:
        """Build a ``MetricSet`` from an existing long-format frame.

        Args:
            df (pd.DataFrame): Long-format rows (with or without ``sim_years``).
            sim_years (float, optional): Override the simulation length. If not
                given it is read from a ``sim_years`` column. Defaults to None.

        Returns:
            MetricSet: The wrapped table.

        Raises:
            ValueError: If ``sim_years`` is neither supplied nor present.
        """
        if sim_years is None:
            if "sim_years" not in df.columns:
                raise ValueError("sim_years must be supplied or present as a column")
            sim_years = float(df["sim_years"].iloc[0])
        return cls(df, sim_years=sim_years)

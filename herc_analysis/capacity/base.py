"""L4 -- ``CapacityBase`` abstract base for capacity-accreditation methods.

A base for market capacity-accreditation methods (MISO today, SPP next).
Subclasses share an hourly frame, per-component resource classes and an
interconnect limit; they differ in the accreditation math and the
price/report tables. ``to_metrics`` emits results in the same long-format
schema as :class:`~herc_analysis.metrics.MetricSet` (with ``scaling="annual"``)
so capacity composes directly into :class:`~herc_analysis.comparison.Comparison`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from herc_analysis.metrics import METRIC_COLUMNS


class CapacityBase(ABC):
    """Base class for market capacity-accreditation methods.

    Args:
        frame (pd.DataFrame): Hourly (or finer) input frame with ``time_utc``
            and one column per component.
        components (list[str]): Component names participating in accreditation.
        classes (list[str]): Resource class per component (same order).
        interconnect_limit_mw (float): Interconnect limit in MW.
        sim_years (float, optional): Simulation length in years, carried into
            ``to_metrics`` so capacity rows convert between annual and total
            views inside ``Comparison``. Defaults to None.
    """

    def __init__(
        self,
        *,
        frame: pd.DataFrame,
        components: list[str],
        classes: list[str],
        interconnect_limit_mw: float,
        sim_years: float | None = None,
    ):
        self.frame = frame
        self.components = components
        self.classes = classes
        self.interconnect_limit_mw = interconnect_limit_mw
        self.sim_years = sim_years

    @abstractmethod
    def accredit(self) -> dict:
        """Seasonal accredited capacity, keyed by ``(season, component)`` (MW)."""

    @abstractmethod
    def revenue(self) -> dict:
        """Annual capacity revenue per component (dollars)."""

    @abstractmethod
    def report(self) -> dict | pd.DataFrame:
        """A tabular summary of the accreditation result."""

    def to_metrics(self) -> pd.DataFrame:
        """Emit capacity revenue in the long-format ``MetricSet`` schema.

        One ``capacity_revenue`` row per component plus a ``plant`` total, all
        tagged ``scaling="annual"`` (capacity revenue is determined by an annual
        auction) so they compose straight into ``Comparison``. A ``sim_years``
        column is included so the annual/total scaling resolves there.

        Returns:
            pd.DataFrame: Long-format rows with :data:`METRIC_COLUMNS` plus
            ``sim_years``.
        """
        rev = self.revenue()

        def _row(entity, value):
            return {
                "entity": entity,
                "metric": "capacity_revenue",
                "resolution": "total",
                "period": "total",
                "value": float(value),
                "unit": "$",
                "scaling": "annual",
                "sim_years": self.sim_years,
            }

        rows = [_row(comp, val) for comp, val in rev.items()]
        rows.append(_row("plant", sum(rev.values())))
        return pd.DataFrame(rows, columns=[*METRIC_COLUMNS, "sim_years"])

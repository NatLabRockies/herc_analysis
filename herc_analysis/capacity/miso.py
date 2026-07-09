"""L4 -- ``MisoCapacity``: MISO DLOL seasonal accredited capacity.

This composes the trusted MISO DLOL engine (now relocated into the capacity
package as ``herc_analysis.capacity._miso_engine``) and re-expresses it as a
:class:`~herc_analysis.capacity.base.CapacityBase` subclass with the design-facing
API (``accredit`` / ``revenue`` / ``report`` / ``to_metrics`` and a
``from_scenario`` constructor that injects availability providers). The
accreditation math is unchanged -- this layer composes the existing, tested
staged pipeline rather than reimplementing it, so MISO outputs match the
pre-refactor code exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from herc_analysis.capacity._miso_engine import MisoCapacity as _LegacyMisoCapacity
from herc_analysis.capacity.base import CapacityBase
from herc_analysis.constants import COMPONENT_TYPE_TO_MISO_CLASS

if TYPE_CHECKING:
    from herc_analysis.scenario import Scenario


class MisoCapacity(CapacityBase):
    """MISO DLOL seasonal accredited capacity.

    Construction mirrors the underlying engine; prefer :meth:`from_scenario`,
    which derives the component/class lists from a ``Scenario`` and injects any
    availability providers.

    Args:
        component_list (list[str]): Component column names in ``df``.
        class_list (list[str]): MISO resource class per component (same order).
        df (pd.DataFrame): Frame with ``time_utc`` and one kW column per
            component.
        zone (int): MISO zone number.
        interconnect_limit (float): Interconnect limit in kW.
        **kwargs: Forwarded to the underlying engine (``pra_years``,
            ``priority_order``, ``remove_low_hour_planning_years``, ``verbose``).
    """

    def __init__(
        self,
        component_list: list[str],
        class_list: list[str],
        df: pd.DataFrame,
        zone: int,
        interconnect_limit: float,
        *,
        sim_years: float | None = None,
        **kwargs,
    ):
        self._engine = _LegacyMisoCapacity(
            component_list, class_list, df, zone, interconnect_limit, **kwargs
        )
        super().__init__(
            frame=df,
            components=list(component_list),
            classes=list(class_list),
            interconnect_limit_mw=self._engine.interconnect_limit_mw,
            sim_years=sim_years,
        )

    # ------------------------------------------------------------------
    # CapacityBase interface
    # ------------------------------------------------------------------

    def accredit(self) -> dict:
        """Seasonal accredited capacity (MW), keyed by ``(season, component)``."""
        return dict(self._engine.sac_mw)

    def revenue(self) -> dict:
        """Annual capacity revenue per component (dollars)."""
        return dict(self._engine.annual_revenue)

    def report(self) -> dict:
        """Structured capacity report (per-component and totals tables)."""
        return self._engine.get_capacity_report()

    # ------------------------------------------------------------------
    # Construction from a Scenario
    # ------------------------------------------------------------------

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario,
        *,
        zone: int,
        pra_years: int | list[int] | None = None,
        availability: dict | None = None,
        priority_order=None,
        remove_low_hour_planning_years: bool = True,
        verbose: bool = False,
    ) -> MisoCapacity:
        """Build a ``MisoCapacity`` from a :class:`Scenario`.

        Resource classes are inferred from each component's Hercules type via
        :data:`COMPONENT_TYPE_TO_MISO_CLASS`; components with no MISO mapping
        (e.g. loads) are skipped. Each component contributes its power channel,
        unless an availability provider is supplied for it, in which case the
        provider's ``{name}__availability`` channel is used instead.

        Args:
            scenario (Scenario): The single-run scenario.
            zone (int): MISO zone number.
            pra_years (int | list[int] | None): PRA auction year(s). Defaults to
                None (most recent available).
            availability (dict, optional): ``{component_name: provider}`` mapping;
                each provider exposes ``attach(df, name)``. Defaults to None.
            priority_order: Interconnect-headroom priority. Defaults to None.
            remove_low_hour_planning_years (bool): Drop sparsely-covered
                planning years. Defaults to True.
            verbose (bool): Engine verbosity. Defaults to False.

        Returns:
            MisoCapacity: The accreditation over the scenario's components.

        Raises:
            ValueError: If no components map to a MISO resource class.
        """
        availability = availability or {}
        channels = scenario.channels

        component_list: list[str] = []
        class_list: list[str] = []
        columns: dict[str, pd.Series] = {}

        for comp in scenario.components:
            miso_class = COMPONENT_TYPE_TO_MISO_CLASS.get(comp.component_type)
            if miso_class is None:
                continue
            component_list.append(comp.name)
            class_list.append(miso_class)
            if comp.name in availability:
                attached = availability[comp.name].attach(channels, comp.name)
                series = attached[f"{comp.name}__availability"]
            else:
                series = channels[f"{comp.name}__power_kw"]
            columns[comp.name] = series.reset_index(drop=True)

        if not component_list:
            raise ValueError(
                "No components map to a MISO resource class; nothing to accredit."
            )

        df = pd.DataFrame({"time_utc": channels["time_utc"].reset_index(drop=True)})
        for name in component_list:
            df[name] = columns[name]

        return cls(
            component_list,
            class_list,
            df,
            zone=zone,
            interconnect_limit=scenario.meta.interconnect_mw * 1000.0,
            sim_years=scenario.meta.sim_years,
            pra_years=pra_years,
            priority_order=priority_order,
            remove_low_hour_planning_years=remove_low_hour_planning_years,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Pass-through to the underlying engine
    # ------------------------------------------------------------------

    #: Engine helpers intentionally exposed on the wrapper (tables, plots,
    #: result dicts and the two staged hourly frames used for diagnostics).
    #: Anything else on the private engine stays private.
    _ENGINE_PASSTHROUGH = frozenset(
        {
            # Tables / printing / getters
            "get_capacity_report",
            "get_component_table",
            "get_totals_table",
            "get_total_revenue",
            "get_component_revenue",
            "print_all_component_tables",
            "print_annual_revenue",
            # Plots
            "plot_sac_stacked_bar",
            "plot_revenue_stacked_bar",
            "plot_isac_availability_bar",
            "plot_isac_hourly_scatter",
            # Result dicts
            "sac_mw",
            "isac_mw",
            "zrc_mw",
            "annual_revenue",
            "revenue_per_season",
            "pra_prices",
            "days_per_season",
            "hours_per_planning_year_season",
            # Staged hourly frames (diagnostics; see example 01)
            "df_h_mw",
            "df_h_limit_mw",
        }
    )

    def __getattr__(self, name: str):
        # Only reached for attributes not found on the wrapper itself; delegate
        # a curated set of engine helpers (plots, tables, result dicts).
        engine = self.__dict__.get("_engine")
        if engine is None or name not in self._ENGINE_PASSTHROUGH:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}; "
                "engine internals are private (use .accredit(), .revenue(), "
                ".report(), or one of the documented helpers)."
            )
        return getattr(engine, name)

"""L2 -- the single-scenario analysis object.

``Scenario`` is the merge the refactor asks for: one object for the analysis of
one Hercules run, replacing the ``OutputAnalysis`` + ``TotalMetrics`` pair.

Construction is cheap (it only discovers components). The two heavy products --
the derived ``channels`` frame and the scalar ``metrics`` -- are ``cached_property``
values assembled from the pure L1 functions, so each stage is unit-testable in
isolation and the raw frame is never copied (it stays on ``self.output``).

Channels-frame convention (see the refactoring plan, sections 6.1/6.4): derived
columns use ``{component}__{signal}`` (double underscore so component names may
themselves contain underscores), plant-level columns use a ``plant__`` prefix,
power stays in Hercules' kW convention (``__power_kw``), energy is MWh
(``__energy_mwh``) and revenue is dollars (``__revenue_*_usd``).

Phase 2 deliberately keeps the *outputs* identical to the current code: the
``metrics`` / ``monthly_metrics`` nested dicts reproduce
``TotalMetrics.compute_metrics`` / ``compute_monthly_metrics`` exactly (the
"no behavior change" contract, verified against the golden snapshots). The
derived channels carry every physical quantity the old ``OutputAnalysis.df``
held, but in the new naming/unit convention rather than as a raw copy; Phase 3
re-expresses ``metrics`` as a long-format ``MetricSet`` (with ``to_nested()``
returning this same dict).
"""

from __future__ import annotations

from functools import cached_property

import pandas as pd
from hercules.hercules_output import HerculesOutput

from herc_analysis import channels, pricing, reducers
from herc_analysis.components import ComponentInfo, discover_components
from herc_analysis.io import RunMeta, load_run
from herc_analysis.metrics import MetricSet


def _ch(name: str, signal: str) -> str:
    """Build a derived channel column name: ``{name}__{signal}``."""
    return f"{name}__{signal}"


# Long-format export specs: (metric, nested-section, nested-key, unit, scaling).
# These map the curated numeric metrics onto the canonical MetricSet schema.
_PLANT_METRIC_SPECS = (
    ("energy_mwh", "plant", "total_energy_mwh", "MWh", "extensive"),
    ("capacity_factor", "plant", "capacity_factor", "-", "intensive"),
    ("revenue_rt", "plant", "total_revenue_rt_k", "k$", "extensive"),
    ("revenue_da", "plant", "total_revenue_da_k", "k$", "extensive"),
    (
        "surplus_capacity_revenue_rt",
        "plant",
        "surplus_capacity_revenue_rt_k",
        "k$",
        "extensive",
    ),
    (
        "surplus_capacity_revenue_da",
        "plant",
        "surplus_capacity_revenue_da_k",
        "k$",
        "extensive",
    ),
    ("avg_price_rt", "market_metrics", "avg_price_rt", "$/MWh", "intensive"),
    ("value_factor", "market_metrics", "value_factor", "-", "intensive"),
)

# (metric, nested-key, unit, scaling) within a component's metric dict.
_COMPONENT_METRIC_SPECS = (
    ("energy_mwh", "energy_mwh", "MWh", "extensive"),
    ("revenue_rt", "revenue_rt_k", "k$", "extensive"),
    ("revenue_da", "revenue_da_k", "k$", "extensive"),
)
_STORAGE_METRIC_SPECS = (
    ("energy_discharge_mwh", "energy_discharge_mwh", "MWh", "extensive"),
    ("energy_charge_mwh", "energy_charge_mwh", "MWh", "extensive"),
    ("battery_mileage_soc", "battery_mileage_soc", "SOC", "extensive"),
    ("battery_mileage_mwh", "battery_mileage_mwh", "MWh", "extensive"),
)


def _rows_from_nested(nested: dict, *, resolution: str, period: str) -> list[dict]:
    """Emit long-format rows for one nested metrics dict at one (resolution, period)."""
    rows: list[dict] = []

    def _row(entity, metric, value, unit, scaling):
        rows.append(
            {
                "entity": entity,
                "metric": metric,
                "resolution": resolution,
                "period": period,
                "value": value,
                "unit": unit,
                "scaling": scaling,
            }
        )

    for metric, section, key, unit, scaling in _PLANT_METRIC_SPECS:
        _row("plant", metric, nested[section][key], unit, scaling)

    for name, cm in nested["components"].items():
        for metric, key, unit, scaling in _COMPONENT_METRIC_SPECS:
            _row(name, metric, cm[key], unit, scaling)
        if cm.get("category") == "storage":
            for metric, key, unit, scaling in _STORAGE_METRIC_SPECS:
                if key in cm:
                    _row(name, metric, cm[key], unit, scaling)

    return rows


class Scenario:
    """Time-series build-up and metrics for a single Hercules run.

    Replaces ``OutputAnalysis`` + ``TotalMetrics``. Construction is cheap; the
    ``channels`` frame and the ``metrics`` / ``monthly_metrics`` sets are cached
    on first access. The raw frame and run metadata are not copied -- they are
    read through ``self.output``.
    """

    def __init__(
        self,
        output: HerculesOutput | str,
        *,
        name: str | None = None,
        resolutions: tuple[str, ...] = ("total", "annual"),
    ):
        """Open or wrap a Hercules run.

        Args:
            output (HerculesOutput | str): An opened ``HerculesOutput`` or a path
                to an H5 file to load.
            name (str, optional): A label for the scenario. Defaults to
                ``"scenario"``.
            resolutions (tuple[str, ...], optional): Temporal resolutions the
                metrics will eventually expose. Retained for the Phase 3
                ``MetricSet`` work; not yet consumed. Defaults to
                ``("total", "annual")``.
        """
        self.output = output if isinstance(output, HerculesOutput) else load_run(output)
        self.meta = RunMeta.from_output(self.output)
        self.name = name or "scenario"
        self.resolutions = resolutions
        self.components: list[ComponentInfo] = discover_components(self.output.h_dict)

    # ------------------------------------------------------------------
    # Convenience views (no copies)
    # ------------------------------------------------------------------

    @property
    def generators(self) -> list[str]:
        """Names of generator components."""
        return [c.name for c in self.components if c.category == "generator"]

    @property
    def storage(self) -> list[str]:
        """Names of storage components."""
        return [c.name for c in self.components if c.category == "storage"]

    @property
    def loads(self) -> list[str]:
        """Names of load components."""
        return [c.name for c in self.components if c.category == "load"]

    def get_component(self, name: str) -> ComponentInfo | None:
        """Look up a :class:`ComponentInfo` by name, or ``None``."""
        for c in self.components:
            if c.name == name:
                return c
        return None

    # ------------------------------------------------------------------
    # Channels frame (built once, cached)
    # ------------------------------------------------------------------

    @cached_property
    def channels(self) -> pd.DataFrame:
        """Derived time-series channels, assembled by composing L1 functions.

        A pipeline of pure stages: base signals -> per-component channels ->
        the single excess-allocation primitive (twice) -> plant aggregates.
        Nothing mutates the raw output.

        Returns:
            pd.DataFrame: The derived channels frame (raw data stays on
            ``self.output``).
        """
        df, work = self._base_frame()
        for comp in self.components:
            self._add_component_channels(df, work, comp)
        self._apply_excess(df)
        self._add_plant_aggregates(df)
        return df

    @cached_property
    def metrics(self) -> dict:
        """Scalar metrics for the whole run as a nested dict.

        Reproduces ``TotalMetrics.compute_metrics(display=False)`` exactly.
        Phase 3 re-expresses this as a long-format ``MetricSet`` whose
        ``to_nested()`` returns the same shape.

        Returns:
            dict: Nested metrics (``simulation_metadata``, ``components``,
            ``categories``, ``plant``, ``market_metrics``).
        """
        return self._scalar_metrics(
            self.channels,
            sim_time_s=self.meta.total_simulation_time_s,
            n_rows=self.meta.n_rows,
            time_key="total_simulation_time_s",
            include_tb4=True,
        )

    @cached_property
    def monthly_metrics(self) -> dict:
        """Per-month scalar metrics, keyed by ``"YYYY-MM"``.

        Reproduces ``TotalMetrics.compute_monthly_metrics(display=False)``.

        Returns:
            dict: Month-string keys mapping to per-month nested metric dicts.

        Raises:
            ValueError: If the channels frame has no ``time_utc`` column.
        """
        df = self.channels
        if "time_utc" not in df.columns:
            raise ValueError(
                "time_utc column is required for monthly metrics computation"
            )
        time_utc = df["time_utc"]
        if getattr(time_utc.dt, "tz", None) is not None:
            # Drop tz before to_period (the naive UTC wall-clock gives identical
            # month buckets) to avoid a noisy PeriodArray tz warning.
            time_utc = time_utc.dt.tz_localize(None)
        month_year = time_utc.dt.to_period("M")
        out: dict = {}
        for month in sorted(month_year.unique()):
            md = df[month_year == month]
            if len(md) == 0:
                continue
            out[str(month)] = self._scalar_metrics(
                md,
                sim_time_s=len(md) * self.meta.dt_s,
                n_rows=len(md),
                time_key="month_simulation_time_s",
                include_tb4=False,
            )
        return out

    @cached_property
    def metric_set(self) -> MetricSet:
        """The run's metrics as a canonical long-format :class:`MetricSet`.

        Carries the curated numeric metrics (plant, per-component, market) at
        ``resolution="total"`` plus, when ``time_utc`` is available, one bucket
        per month (``resolution="monthly"``). This is the shape ``Comparison``
        consumes and what ``MetricSet.to_csv`` writes.

        Returns:
            MetricSet: Long-format metrics for this run.
        """
        rows = _rows_from_nested(self.metrics, resolution="total", period="total")
        if "time_utc" in self.channels.columns:
            for month, mm in self.monthly_metrics.items():
                rows.extend(_rows_from_nested(mm, resolution="monthly", period=month))
        return MetricSet(pd.DataFrame(rows), sim_years=self.meta.sim_years)

    # ------------------------------------------------------------------
    # Channel build stages (private)
    # ------------------------------------------------------------------

    def _base_frame(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build the base frame and return it with the deduped raw work frame.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: ``(channels_base, work)`` where
            ``work`` is a deduped copy of the raw frame, passed to the
            per-component stage and then discarded (never retained on ``self``).
        """
        work = self.output.df.copy()
        work["time"] = work["time"].astype(float)
        if work["time"].nunique() != len(work):
            work = work.drop_duplicates(subset=["time"], keep="first")
        work = work.reset_index(drop=True)

        df = pd.DataFrame(index=work.index)
        df["time"] = work["time"]
        if "time_utc" in work.columns:
            df["time_utc"] = work["time_utc"]

        lmp_rt, lmp_da = self._extract_lmp(work)
        df["lmp_rt"] = lmp_rt
        df["lmp_da"] = lmp_da

        if "time_utc" in df.columns:
            df["lmp_rt_hourly"] = channels.hourly_mean(df["lmp_rt"], df["time_utc"])
        else:
            df["lmp_rt_hourly"] = (
                df["lmp_rt"].groupby(df["time"] // 3600).transform("mean")
            )

        if "plant.locally_generated_power" in work.columns:
            df["plant__local_gen_mw"] = channels.kw_to_mw(
                work["plant.locally_generated_power"]
            )
        return df, work

    def _extract_lmp(self, work: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Extract RT/DA LMP series, mirroring ``_process_external_signals``."""
        zeros = pd.Series(0.0, index=work.index)
        if not self.output.h_dict.get("external_signals"):
            return zeros, zeros.copy()

        lmp_rt_col = lmp_da_col = None
        for col in work.columns:
            if "lmp_rt" in col.lower():
                lmp_rt_col = col
            elif "lmp_da" in col.lower():
                lmp_da_col = col

        if lmp_rt_col and lmp_da_col:
            return work[lmp_rt_col], work[lmp_da_col]
        return zeros, zeros.copy()

    def _add_component_channels(
        self, df: pd.DataFrame, work: pd.DataFrame, comp: ComponentInfo
    ) -> None:
        """Add power/energy/revenue (and storage/generator extras) for one comp."""
        name = comp.name

        power_col = f"{name}.power"
        if power_col in work.columns:
            df[_ch(name, "power_kw")] = work[power_col]
        else:
            df[_ch(name, "power_kw")] = 0.0

        setpoint_col = f"{name}.power_setpoint"
        if setpoint_col in work.columns:
            df[_ch(name, "power_setpoint_kw")] = work[setpoint_col]

        energy = channels.energy_mwh(df[_ch(name, "power_kw")], self.meta.dt_s)
        df[_ch(name, "energy_mwh")] = energy
        df[_ch(name, "revenue_rt_usd")] = channels.revenue_usd(df["lmp_rt"], energy)
        df[_ch(name, "revenue_da_usd")] = channels.revenue_usd(df["lmp_da"], energy)

        # Baselines (never modified by the excess corrections).
        df[_ch(name, "revenue_rt_if_all_paid_usd")] = df[_ch(name, "revenue_rt_usd")]
        df[_ch(name, "revenue_da_if_all_paid_usd")] = df[_ch(name, "revenue_da_usd")]

        if comp.category == "generator":
            df[_ch(name, "excess_local_generation_mw")] = 0.0
            df[_ch(name, "revenue_rt_excess_loss_usd")] = 0.0
            df[_ch(name, "revenue_da_excess_loss_usd")] = 0.0

        if comp.category == "storage":
            rev_rt = df[_ch(name, "revenue_rt_usd")]
            rev_da = df[_ch(name, "revenue_da_usd")]
            df[_ch(name, "revenue_rt_discharge_usd")] = rev_rt.where(energy > 0, 0.0)
            df[_ch(name, "revenue_rt_charge_usd")] = rev_rt.where(energy < 0, 0.0)
            df[_ch(name, "revenue_da_discharge_usd")] = rev_da.where(energy > 0, 0.0)
            df[_ch(name, "revenue_da_charge_usd")] = rev_da.where(energy < 0, 0.0)

            df[_ch(name, "revenue_rt_charge_if_all_paid_usd")] = df[
                _ch(name, "revenue_rt_charge_usd")
            ]
            df[_ch(name, "revenue_da_charge_if_all_paid_usd")] = df[
                _ch(name, "revenue_da_charge_usd")
            ]

            df[_ch(name, "excess_absorbed_mw")] = 0.0
            df[_ch(name, "revenue_rt_charge_savings_usd")] = 0.0
            df[_ch(name, "revenue_da_charge_savings_usd")] = 0.0

            soc_col = f"{name}.soc"
            df[_ch(name, "soc")] = work[soc_col] if soc_col in work.columns else 0.0

    def _apply_excess(self, df: pd.DataFrame) -> None:
        """Apply both excess corrections via the single allocation primitive."""
        if "plant__local_gen_mw" in df.columns:
            excess = pricing.excess_over_limit(
                df["plant__local_gen_mw"], self.meta.interconnect_mw
            )
        else:
            excess = pd.Series(0.0, index=df.index)
        df["plant__excess_local_generation_mw"] = excess

        energy_per_mw = self.meta.dt_s / 3600.0
        has_excess = bool((excess > 0).any())

        # Storage absorbs excess (capped at each unit's charging power).
        if self.storage and has_excess:
            charge_mw = {
                n: channels.kw_to_mw((-df[_ch(n, "power_kw")]).clip(lower=0))
                for n in self.storage
            }
            absorbed = pricing.allocate_proportional(
                excess, charge_mw, cap_to_weight=True
            )
            for n in self.storage:
                absorbed_mw = absorbed[n]
                savings_rt = df["lmp_rt"] * absorbed_mw * energy_per_mw
                savings_da = df["lmp_da"] * absorbed_mw * energy_per_mw
                df[_ch(n, "excess_absorbed_mw")] = absorbed_mw
                df[_ch(n, "revenue_rt_charge_savings_usd")] = savings_rt
                df[_ch(n, "revenue_da_charge_savings_usd")] = savings_da
                df[_ch(n, "revenue_rt_charge_usd")] = (
                    df[_ch(n, "revenue_rt_charge_usd")] + savings_rt
                )
                df[_ch(n, "revenue_da_charge_usd")] = (
                    df[_ch(n, "revenue_da_charge_usd")] + savings_da
                )
                df[_ch(n, "revenue_rt_usd")] = df[_ch(n, "revenue_rt_usd")] + savings_rt
                df[_ch(n, "revenue_da_usd")] = df[_ch(n, "revenue_da_usd")] + savings_da

        # Generators lose revenue on the un-delivered excess (power share).
        if self.generators and has_excess:
            gen_mw = {
                n: channels.kw_to_mw(df[_ch(n, "power_kw")].clip(lower=0))
                for n in self.generators
            }
            gen_excess = pricing.allocate_proportional(excess, gen_mw)
            for n in self.generators:
                gen_excess_mw = gen_excess[n]
                loss_rt = df["lmp_rt"] * gen_excess_mw * energy_per_mw
                loss_da = df["lmp_da"] * gen_excess_mw * energy_per_mw
                df[_ch(n, "excess_local_generation_mw")] = gen_excess_mw
                df[_ch(n, "revenue_rt_excess_loss_usd")] = loss_rt
                df[_ch(n, "revenue_da_excess_loss_usd")] = loss_da
                df[_ch(n, "revenue_rt_usd")] = df[_ch(n, "revenue_rt_usd")] - loss_rt
                df[_ch(n, "revenue_da_usd")] = df[_ch(n, "revenue_da_usd")] - loss_da

    def _add_plant_aggregates(self, df: pd.DataFrame) -> None:
        """Add plant total power and the surplus / ideal capacity columns."""
        power_mw_cols = [
            channels.kw_to_mw(df[_ch(c.name, "power_kw")])
            for c in self.components
            if _ch(c.name, "power_kw") in df.columns
        ]
        total_power_mw = (
            sum(power_mw_cols) if power_mw_cols else pd.Series(0.0, index=df.index)
        )
        df["plant__total_power_mw"] = total_power_mw

        ic = self.meta.interconnect_mw
        energy_per_mw = self.meta.dt_s / 3600.0

        surplus_mw = pricing.surplus_capacity_mw(ic, total_power_mw)
        surplus_energy = surplus_mw * energy_per_mw
        df["plant__surplus_capacity_mw"] = surplus_mw
        df["plant__surplus_capacity_energy_mwh"] = surplus_energy
        df["plant__surplus_capacity_revenue_rt_usd"] = surplus_energy * df["lmp_rt"]
        df["plant__surplus_capacity_revenue_da_usd"] = surplus_energy * df["lmp_da"]

        for mkt in ("rt", "da"):
            pos, neg = pricing.capacity_revenue_by_sign(
                surplus_energy, df[f"lmp_{mkt}"]
            )
            df[f"plant__surplus_capacity_revenue_{mkt}_positive_lmp_usd"] = pos
            df[f"plant__surplus_capacity_revenue_{mkt}_negative_lmp_usd"] = neg

        ideal_energy = pd.Series(ic * energy_per_mw, index=df.index)
        for mkt in ("rt", "da"):
            pos, neg = pricing.capacity_revenue_by_sign(ideal_energy, df[f"lmp_{mkt}"])
            df[f"plant__ideal_capacity_revenue_{mkt}_positive_lmp_usd"] = pos
            df[f"plant__ideal_capacity_revenue_{mkt}_negative_lmp_usd"] = neg

    # ------------------------------------------------------------------
    # Metric computation (private)
    # ------------------------------------------------------------------

    def _scalar_metrics(
        self,
        df: pd.DataFrame,
        *,
        sim_time_s: float,
        n_rows: int,
        time_key: str,
        include_tb4: bool,
    ) -> dict:
        """Compute the nested metrics dict for ``df`` (whole run or one month)."""
        meta = self.meta
        metrics: dict = {
            "simulation_metadata": {
                time_key: sim_time_s,
                "interconnect_mw": meta.interconnect_mw,
                "dt": meta.dt_s,
                "n_rows": n_rows,
            },
            "components": {},
            "categories": {},
            "plant": {},
            "market_metrics": {},
        }

        total_energy = 0.0
        total_rev_rt = 0.0
        total_rev_da = 0.0

        for comp in self.components:
            name = comp.name
            energy = reducers.total(df[_ch(name, "energy_mwh")])
            rev_rt = reducers.total(df[_ch(name, "revenue_rt_usd")])
            rev_da = reducers.total(df[_ch(name, "revenue_da_usd")])

            total_energy += energy
            total_rev_rt += rev_rt
            total_rev_da += rev_da

            cm: dict = {
                "component_type": comp.component_type,
                "category": comp.category,
                "energy_mwh": energy,
                "revenue_rt_k": rev_rt / 1e3,
                "revenue_da_k": rev_da / 1e3,
                "revenue_rt_if_all_paid_k": reducers.total(
                    df[_ch(name, "revenue_rt_if_all_paid_usd")]
                )
                / 1e3,
                "revenue_da_if_all_paid_k": reducers.total(
                    df[_ch(name, "revenue_da_if_all_paid_usd")]
                )
                / 1e3,
            }

            if comp.category == "generator":
                cm["revenue_rt_excess_loss_k"] = (
                    reducers.total(df[_ch(name, "revenue_rt_excess_loss_usd")]) / 1e3
                )
                cm["revenue_da_excess_loss_k"] = (
                    reducers.total(df[_ch(name, "revenue_da_excess_loss_usd")]) / 1e3
                )

            if comp.category == "storage":
                cm.update(self._storage_metrics(df, name, include_tb4=include_tb4))

            metrics["components"][name] = cm

        for category, names in [
            ("generator", self.generators),
            ("storage", self.storage),
            ("load", self.loads),
        ]:
            if not names:
                continue
            metrics["categories"][category] = {
                "energy_mwh": sum(
                    metrics["components"][n]["energy_mwh"] for n in names
                ),
                "revenue_rt_k": sum(
                    metrics["components"][n]["revenue_rt_k"] for n in names
                ),
                "revenue_da_k": sum(
                    metrics["components"][n]["revenue_da_k"] for n in names
                ),
            }

        sim_hours = sim_time_s / 3600.0
        metrics["plant"]["total_energy_mwh"] = total_energy
        metrics["plant"]["capacity_factor"] = reducers.capacity_factor(
            total_energy, meta.interconnect_mw, sim_hours
        )
        metrics["plant"]["total_revenue_rt_k"] = total_rev_rt / 1e3
        metrics["plant"]["total_revenue_da_k"] = total_rev_da / 1e3
        metrics["plant"].update(self._surplus_metrics(df))

        avg_price_rt = reducers.mean(df["lmp_rt"])
        avg_price_revenue = avg_price_rt * total_energy
        metrics["market_metrics"]["avg_price_rt"] = avg_price_rt
        metrics["market_metrics"]["avg_price_revenue_k"] = avg_price_revenue / 1e3
        metrics["market_metrics"]["value_factor"] = reducers.value_factor(
            total_rev_rt, avg_price_rt, total_energy
        )

        return metrics

    def _storage_metrics(
        self, df: pd.DataFrame, name: str, *, include_tb4: bool
    ) -> dict:
        """Storage-specific metric entries (splits, savings, TB4, mileage)."""
        energy = df[_ch(name, "energy_mwh")]
        out: dict = {
            "energy_discharge_mwh": reducers.total(energy.where(energy > 0, 0.0)),
            "energy_charge_mwh": reducers.total(energy.where(energy < 0, 0.0)),
            "revenue_rt_discharge_k": reducers.total(
                df[_ch(name, "revenue_rt_discharge_usd")]
            )
            / 1e3,
            "revenue_rt_charge_k": reducers.total(
                df[_ch(name, "revenue_rt_charge_usd")]
            )
            / 1e3,
            "revenue_rt_charge_if_all_paid_k": reducers.total(
                df[_ch(name, "revenue_rt_charge_if_all_paid_usd")]
            )
            / 1e3,
            "revenue_rt_charge_savings_k": reducers.total(
                df[_ch(name, "revenue_rt_charge_savings_usd")]
            )
            / 1e3,
            "revenue_da_charge_k": reducers.total(
                df[_ch(name, "revenue_da_charge_usd")]
            )
            / 1e3,
            "revenue_da_charge_if_all_paid_k": reducers.total(
                df[_ch(name, "revenue_da_charge_if_all_paid_usd")]
            )
            / 1e3,
            "revenue_da_charge_savings_k": reducers.total(
                df[_ch(name, "revenue_da_charge_savings_usd")]
            )
            / 1e3,
        }
        if include_tb4:
            out.update(self._tb4_metrics(df, name))
        out.update(self._mileage_metrics(df, name))
        return out

    def _tb4_metrics(self, df: pd.DataFrame, name: str) -> dict:
        """TB4 (top/bottom 4-hour price spread) revenue for a storage unit."""
        n_pts_4h = int(4 * 3600 / self.meta.dt_s)
        energy_cap_mwh = (
            self.output.h_dict.get(name, {}).get("energy_capacity", 0) / 1000.0
        )

        if "time_utc" in df.columns:
            grp = df.groupby(df["time_utc"].dt.date)
        else:
            grp = df.groupby(df["time"] // 86400)

        daily_tb4_rt = grp.apply(
            lambda x: reducers.tb4_spread(x["lmp_rt_hourly"], n_pts_4h)
        )
        daily_tb4_da = grp.apply(lambda x: reducers.tb4_spread(x["lmp_da"], n_pts_4h))
        return {
            "optimum_tb4_revenue_rt_k": daily_tb4_rt.sum() * energy_cap_mwh / 1e3,
            "optimum_tb4_revenue_da_k": daily_tb4_da.sum() * energy_cap_mwh / 1e3,
        }

    def _mileage_metrics(self, df: pd.DataFrame, name: str) -> dict:
        """Battery mileage (absolute SOC distance) in SOC and energy units."""
        soc_col = _ch(name, "soc")
        if soc_col not in df.columns:
            return {"battery_mileage_soc": 0.0, "battery_mileage_mwh": 0.0}
        mileage_soc = channels.soc_mileage(df[soc_col])
        energy_cap_mwh = (
            self.output.h_dict.get(name, {}).get("energy_capacity", 0) / 1000.0
        )
        return {
            "battery_mileage_soc": mileage_soc,
            "battery_mileage_mwh": mileage_soc * energy_cap_mwh,
        }

    @staticmethod
    def _surplus_metrics(df: pd.DataFrame) -> dict:
        """Surplus and ideal capacity revenue metrics (plant-level)."""
        out: dict = {}
        mask = df["plant__surplus_capacity_mw"] > 0
        out["surplus_capacity_revenue_rt_k"] = (
            df.loc[mask, "plant__surplus_capacity_revenue_rt_usd"].sum() / 1e3
        )
        out["surplus_capacity_revenue_da_k"] = (
            df.loc[mask, "plant__surplus_capacity_revenue_da_usd"].sum() / 1e3
        )
        for tag in ("positive", "negative"):
            for mkt in ("rt", "da"):
                scol = f"plant__surplus_capacity_revenue_{mkt}_{tag}_lmp_usd"
                out[f"surplus_capacity_revenue_{mkt}_{tag}_lmp_k"] = (
                    df[scol].sum() / 1e3
                )
                icol = f"plant__ideal_capacity_revenue_{mkt}_{tag}_lmp_usd"
                out[f"ideal_capacity_revenue_{mkt}_{tag}_lmp_k"] = df[icol].sum() / 1e3
        return out

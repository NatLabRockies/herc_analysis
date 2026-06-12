"""TotalMetrics class for computing and comparing Hercules simulation metrics."""

import json
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class TotalMetrics:
    """Compute and compare scalar and monthly metrics from Hercules simulations.

    All metrics are computed generically per-component, aggregated by category
    (generator / storage / load), and summed at the plant level.
    """

    _PCT_CHANGE_COLUMN = "% Change"

    def __init__(
        self,
        output_analysis,
        scenario_names: list[str] | None = None,
    ):
        """Initialize TotalMetrics with one or more OutputAnalysis objects.

        Args:
            output_analysis (OutputAnalysis | list[OutputAnalysis]): Single
                OutputAnalysis or list for multi-scenario comparison.
            scenario_names (list[str], optional): Names for each scenario.
                Defaults to None.
        """
        if isinstance(output_analysis, list):
            self.output_analyses = output_analysis
            self.is_multi_scenario = True
            if scenario_names is None:
                self.scenario_names = [
                    f"Scenario {i + 1}" for i in range(len(output_analysis))
                ]
            else:
                if len(scenario_names) != len(output_analysis):
                    raise ValueError(
                        "scenario_names length must match output_analysis list length"
                    )
                self.scenario_names = scenario_names
        else:
            self.output_analyses = [output_analysis]
            self.is_multi_scenario = False
            self.scenario_names = ["Scenario 1"]

        self.metrics = None
        self.monthly_metrics = None

    # ------------------------------------------------------------------
    # Total (scalar) metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, display: bool = True) -> dict | list[dict]:
        """Compute total energy, capacity, revenue, and market metrics.

        Args:
            display (bool, optional): Whether to print a formatted summary.
                Defaults to True.

        Returns:
            dict | list[dict]: Metrics dict (single) or list (multi-scenario).
        """
        all_metrics = [
            self._compute_single_scenario_metrics(oa) for oa in self.output_analyses
        ]
        self.metrics = all_metrics if self.is_multi_scenario else all_metrics[0]

        if display:
            if self.is_multi_scenario:
                for name, m in zip(self.scenario_names, all_metrics, strict=False):
                    print(f"\n{'=' * 55}")
                    print(f"SCENARIO: {name}")
                    self._display_summary(m)
            else:
                self._display_summary(all_metrics[0])

        return self.metrics

    def _compute_single_scenario_metrics(self, oa) -> dict:
        """Compute metrics for a single OutputAnalysis.

        Args:
            oa: OutputAnalysis object.

        Returns:
            dict: Nested metrics dictionary.
        """
        df = oa.df

        metrics: dict = {
            "simulation_metadata": {
                "total_simulation_time_s": oa.total_simulation_time_s,
                "interconnect_mw": oa.interconnect_mw,
                "dt": oa.dt,
                "n_rows": oa.n_rows,
            },
            "components": {},
            "categories": {},
            "plant": {},
            "market_metrics": {},
        }

        total_plant_energy = 0.0
        total_revenue_rt = 0.0
        total_revenue_da = 0.0

        # --- per-component metrics ---
        for comp in oa.components:
            name = comp.name
            energy = df[f"{name}_energy_mwh"].sum()
            rev_rt = df[f"{name}_revenue_rt"].sum()
            rev_da = df[f"{name}_revenue_da"].sum()

            total_plant_energy += energy
            total_revenue_rt += rev_rt
            total_revenue_da += rev_da

            comp_metrics: dict = {
                "component_type": comp.component_type,
                "category": comp.category,
                "energy_mwh": energy,
                "revenue_rt_k": rev_rt / 1e3,
                "revenue_da_k": rev_da / 1e3,
                "revenue_rt_if_all_paid_k": (
                    df[f"{name}_revenue_rt_if_all_paid"].sum() / 1e3
                ),
                "revenue_da_if_all_paid_k": (
                    df[f"{name}_revenue_da_if_all_paid"].sum() / 1e3
                ),
            }

            if comp.category == "generator":
                comp_metrics["revenue_rt_excess_loss_k"] = (
                    df[f"{name}_revenue_rt_excess_loss"].sum() / 1e3
                )
                comp_metrics["revenue_da_excess_loss_k"] = (
                    df[f"{name}_revenue_da_excess_loss"].sum() / 1e3
                )

            if comp.category == "storage":
                discharge = df[df[f"{name}_energy_mwh"] > 0][f"{name}_energy_mwh"].sum()
                charge = df[df[f"{name}_energy_mwh"] < 0][f"{name}_energy_mwh"].sum()
                comp_metrics["energy_discharge_mwh"] = discharge
                comp_metrics["energy_charge_mwh"] = charge
                comp_metrics["revenue_rt_discharge_k"] = (
                    df[f"{name}_revenue_rt_discharge"].sum() / 1e3
                )
                comp_metrics["revenue_rt_charge_k"] = (
                    df[f"{name}_revenue_rt_charge"].sum() / 1e3
                )
                comp_metrics["revenue_rt_charge_if_all_paid_k"] = (
                    df[f"{name}_revenue_rt_charge_if_all_paid"].sum() / 1e3
                )
                comp_metrics["revenue_rt_charge_savings_k"] = (
                    df[f"{name}_revenue_rt_charge_savings"].sum() / 1e3
                )
                comp_metrics["revenue_da_charge_k"] = (
                    df[f"{name}_revenue_da_charge"].sum() / 1e3
                )
                comp_metrics["revenue_da_charge_if_all_paid_k"] = (
                    df[f"{name}_revenue_da_charge_if_all_paid"].sum() / 1e3
                )
                comp_metrics["revenue_da_charge_savings_k"] = (
                    df[f"{name}_revenue_da_charge_savings"].sum() / 1e3
                )

                comp_metrics.update(self._compute_tb4_metrics(df, oa, name))
                comp_metrics.update(self._compute_mileage_metrics(df, oa, name))

            metrics["components"][name] = comp_metrics

        # --- per-category aggregates ---
        for category, names in [
            ("generator", oa.generators),
            ("storage", oa.storage),
            ("load", oa.loads),
        ]:
            if not names:
                continue
            cat_energy = sum(metrics["components"][n]["energy_mwh"] for n in names)
            cat_rev_rt = sum(metrics["components"][n]["revenue_rt_k"] for n in names)
            cat_rev_da = sum(metrics["components"][n]["revenue_da_k"] for n in names)
            metrics["categories"][category] = {
                "energy_mwh": cat_energy,
                "revenue_rt_k": cat_rev_rt,
                "revenue_da_k": cat_rev_da,
            }

        # --- plant-level metrics ---
        sim_hours = oa.total_simulation_time_s / 3600
        capacity_factor_plant = (
            total_plant_energy / (oa.interconnect_mw * sim_hours)
            if oa.interconnect_mw * sim_hours > 0
            else float("nan")
        )

        metrics["plant"]["total_energy_mwh"] = total_plant_energy
        metrics["plant"]["capacity_factor"] = capacity_factor_plant
        metrics["plant"]["total_revenue_rt_k"] = total_revenue_rt / 1e3
        metrics["plant"]["total_revenue_da_k"] = total_revenue_da / 1e3

        # Surplus / ideal capacity
        metrics["plant"].update(self._surplus_metrics(df))

        # Market metrics
        avg_price_rt = df["lmp_rt"].mean()
        avg_price_revenue = avg_price_rt * total_plant_energy
        value_factor = (
            total_revenue_rt / avg_price_revenue
            if total_revenue_rt > 0 and avg_price_revenue > 0
            else float("nan")
        )
        metrics["market_metrics"]["avg_price_rt"] = avg_price_rt
        metrics["market_metrics"]["avg_price_revenue_k"] = avg_price_revenue / 1e3
        metrics["market_metrics"]["value_factor"] = value_factor

        return metrics

    # ------------------------------------------------------------------
    # Storage-specific helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_tb4_metrics(df, oa, name: str) -> dict:
        """Compute TB4 (top/bottom 4-hour spread) metrics for a storage component.

        Args:
            df (pd.DataFrame): Scenario dataframe.
            oa: OutputAnalysis object.
            name (str): Component name.

        Returns:
            dict: TB4 metric entries.
        """
        n_pts_4h = int(4 * 3600 / oa.dt_log)
        comp_dict = oa.h_dict.get(name, {})
        # energy_capacity is stored in kWh in the Hercules H5 dict;
        # convert to MWh for the revenue calculation below.
        energy_cap_mwh = comp_dict.get("energy_capacity", 0) / 1000.0

        if "time_utc" in df.columns:
            grp = df.groupby(df["time_utc"].dt.date)
        else:
            grp = df.groupby(df["time"] // 86400)

        daily_tb4_rt = grp.apply(
            lambda x: (
                x["lmp_rt_hourly"].nlargest(n_pts_4h).mean()
                - x["lmp_rt_hourly"].nsmallest(n_pts_4h).mean()
            )
        )
        daily_tb4_da = grp.apply(
            lambda x: (
                x["lmp_da"].nlargest(n_pts_4h).mean()
                - x["lmp_da"].nsmallest(n_pts_4h).mean()
            )
        )

        return {
            "optimum_tb4_revenue_rt_k": daily_tb4_rt.sum() * energy_cap_mwh / 1e3,
            "optimum_tb4_revenue_da_k": daily_tb4_da.sum() * energy_cap_mwh / 1e3,
        }

    @staticmethod
    def _compute_mileage_metrics(df, oa, name: str) -> dict:
        """Compute battery mileage (total absolute SOC distance travelled).

        Mileage is the cumulative absolute change in state of charge over the
        simulation, a proxy for cycling / throughput. It is reported both in
        SOC units (sum of absolute step-to-step SOC differences) and in energy
        units (SOC mileage scaled by the energy capacity).

        Args:
            df (pd.DataFrame): Scenario dataframe.
            oa: OutputAnalysis object.
            name (str): Component name.

        Returns:
            dict: Mileage metric entries (``battery_mileage_soc`` and
                ``battery_mileage_mwh``).
        """
        soc_col = f"{name}_soc"
        if soc_col not in df.columns:
            return {"battery_mileage_soc": 0.0, "battery_mileage_mwh": 0.0}

        mileage_soc = float(df[soc_col].diff().abs().sum())
        # energy_capacity is stored in kWh in the Hercules H5 dict; convert to
        # MWh so the energy-units mileage is consistent with energy_mwh.
        energy_cap_mwh = oa.h_dict.get(name, {}).get("energy_capacity", 0) / 1000.0

        return {
            "battery_mileage_soc": mileage_soc,
            "battery_mileage_mwh": mileage_soc * energy_cap_mwh,
        }

    # ------------------------------------------------------------------
    # Surplus / ideal capacity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _surplus_metrics(df) -> dict:
        """Compute surplus and ideal capacity revenue metrics.

        Args:
            df (pd.DataFrame): Dataframe with plant-level columns.

        Returns:
            dict: Surplus and ideal capacity metric entries.
        """
        out: dict = {}

        surplus_rev_rt = df[df["surplus_capacity_mw"] > 0][
            "surplus_capacity_revenue_rt"
        ].sum()
        surplus_rev_da = df[df["surplus_capacity_mw"] > 0][
            "surplus_capacity_revenue_da"
        ].sum()
        out["surplus_capacity_revenue_rt_k"] = surplus_rev_rt / 1e3
        out["surplus_capacity_revenue_da_k"] = surplus_rev_da / 1e3

        for tag in ("positive", "negative"):
            for mkt in ("rt", "da"):
                col = f"surplus_capacity_revenue_{mkt}_{tag}_lmp"
                out[f"surplus_capacity_revenue_{mkt}_{tag}_lmp_k"] = df[col].sum() / 1e3
                icol = f"ideal_capacity_revenue_{mkt}_{tag}_lmp"
                out[f"ideal_capacity_revenue_{mkt}_{tag}_lmp_k"] = df[icol].sum() / 1e3

        return out

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _display_summary(self, metrics: dict):
        """Print a formatted summary of metrics.

        Args:
            metrics (dict): Metrics dictionary.
        """
        w = 55
        print("\n" + "=" * w)
        print("Component Energy Production (MWh)")
        for name, cm in metrics["components"].items():
            label = f"  {name} ({cm['category']})"
            print(f"{label:<35}{cm['energy_mwh']:>12,.2f}")
            if cm["category"] == "storage":
                print(
                    f"    discharge / charge:             "
                    f"{cm['energy_discharge_mwh']:>12,.2f} / "
                    f"{cm['energy_charge_mwh']:>,.2f}"
                )
                if "battery_mileage_soc" in cm:
                    print(
                        f"    mileage (SOC / MWh):            "
                        f"{cm['battery_mileage_soc']:>12,.2f} / "
                        f"{cm['battery_mileage_mwh']:>,.2f}"
                    )
        print(f"{'  Plant Total':<35}{metrics['plant']['total_energy_mwh']:>12,.2f}")

        print("=" * w)
        print("Capacity Factor")
        print(f"{'  Plant CF':<35}{metrics['plant']['capacity_factor']:>12.1%}")

        print("=" * w)
        print("Revenue (Thousand $)")
        for name, cm in metrics["components"].items():
            print(
                f"  {name} RT / DA:  {cm['revenue_rt_k']:>10.1f} / {cm['revenue_da_k']:.1f}"
            )
            if cm["category"] == "generator" and "revenue_rt_excess_loss_k" in cm:
                print(
                    f"    RT (paid / if all paid / excess loss): "
                    f"{cm['revenue_rt_k']:>8.1f} / "
                    f"{cm['revenue_rt_if_all_paid_k']:.1f} / "
                    f"{cm['revenue_rt_excess_loss_k']:.1f}"
                )
                print(
                    f"    DA (paid / if all paid / excess loss): "
                    f"{cm['revenue_da_k']:>8.1f} / "
                    f"{cm['revenue_da_if_all_paid_k']:.1f} / "
                    f"{cm['revenue_da_excess_loss_k']:.1f}"
                )
            if cm["category"] == "storage":
                print(
                    f"    discharge / charge RT:  "
                    f"{cm['revenue_rt_discharge_k']:>10.1f} / "
                    f"{cm['revenue_rt_charge_k']:.1f}"
                )
                if "revenue_rt_charge_if_all_paid_k" in cm:
                    print(
                        f"    charge RT (paid / if all paid / savings): "
                        f"{cm['revenue_rt_charge_k']:>8.1f} / "
                        f"{cm['revenue_rt_charge_if_all_paid_k']:.1f} / "
                        f"{cm['revenue_rt_charge_savings_k']:.1f}"
                    )
                    print(
                        f"    charge DA (paid / if all paid / savings): "
                        f"{cm['revenue_da_charge_k']:>8.1f} / "
                        f"{cm['revenue_da_charge_if_all_paid_k']:.1f} / "
                        f"{cm['revenue_da_charge_savings_k']:.1f}"
                    )
                if "optimum_tb4_revenue_rt_k" in cm:
                    print(
                        f"    optimum TB4 RT / DA:   "
                        f"{cm['optimum_tb4_revenue_rt_k']:>10.1f} / "
                        f"{cm['optimum_tb4_revenue_da_k']:.1f}"
                    )
        print(
            f"  Total RT / DA:   "
            f"{metrics['plant']['total_revenue_rt_k']:>10.1f} / "
            f"{metrics['plant']['total_revenue_da_k']:.1f}"
        )

        print("=" * w)
        p = metrics["plant"]
        print("Ideal Capacity Revenue (Thousand $)")
        print(
            f"  RT pos / neg LMP: "
            f"{p['ideal_capacity_revenue_rt_positive_lmp_k']:>10.1f} / "
            f"{p['ideal_capacity_revenue_rt_negative_lmp_k']:.1f}"
        )
        pct = (
            p["total_revenue_rt_k"]
            / p["ideal_capacity_revenue_rt_positive_lmp_k"]
            * 100
            if p["ideal_capacity_revenue_rt_positive_lmp_k"] > 0
            else float("nan")
        )
        print(f"  % Ideal captured RT: {pct:>10.1f}%")

        print("=" * w)
        print("Surplus Capacity Revenue (Thousand $)")
        print(
            f"  RT pos / neg LMP: "
            f"{p['surplus_capacity_revenue_rt_positive_lmp_k']:>10.1f} / "
            f"{p['surplus_capacity_revenue_rt_negative_lmp_k']:.1f}"
        )
        print(
            f"  Total RT / DA:    "
            f"{p['surplus_capacity_revenue_rt_k']:>10.1f} / "
            f"{p['surplus_capacity_revenue_da_k']:.1f}"
        )

        print("=" * w)
        mm = metrics["market_metrics"]
        print(f"Avg Price RT:      {mm['avg_price_rt']:>12.2f} $/MWh")
        print(f"Value Factor:      {mm['value_factor']:>12.3f}")
        print("=" * w)

    # ------------------------------------------------------------------
    # Monthly metrics
    # ------------------------------------------------------------------

    def compute_monthly_metrics(self, display: bool = True) -> dict | list[dict]:
        """Compute monthly metrics for one or more scenarios.

        Args:
            display (bool, optional): Whether to display formatted summary.
                Defaults to True.

        Returns:
            dict | list[dict]: Monthly metrics.
        """
        all_monthly = []
        for oa in self.output_analyses:
            monthly = self._compute_single_scenario_monthly_metrics(oa)
            all_monthly.append({"monthly": monthly})

        self.monthly_metrics = all_monthly if self.is_multi_scenario else all_monthly[0]

        if display:
            if self.is_multi_scenario:
                for name, md in zip(self.scenario_names, all_monthly, strict=False):
                    print(f"\n{'=' * 80}")
                    print(f"SCENARIO: {name}")
                    self._display_monthly_summary(md["monthly"])
            else:
                self._display_monthly_summary(all_monthly[0]["monthly"])

        return self.monthly_metrics

    def _compute_single_scenario_monthly_metrics(self, oa) -> dict:
        """Compute monthly metrics for a single OutputAnalysis.

        Args:
            oa: OutputAnalysis object.

        Returns:
            dict: Month-string keys mapping to per-month metric dicts.
        """
        df = oa.df
        if "time_utc" not in df.columns:
            raise ValueError(
                "time_utc column is required for monthly metrics computation"
            )
        if not pd.api.types.is_datetime64_any_dtype(df["time_utc"]):
            df = df.copy()
            df["time_utc"] = pd.to_datetime(df["time_utc"])

        df_temp = df.copy()
        df_temp["month_year"] = df_temp["time_utc"].dt.to_period("M")
        months = sorted(df_temp["month_year"].unique())

        monthly_metrics: dict = {}
        for month in months:
            month_str = str(month)
            md = df_temp[df_temp["month_year"] == month]
            if len(md) == 0:
                continue

            sim_time_s = len(md) * oa.dt
            sim_hours = sim_time_s / 3600

            mm: dict = {
                "simulation_metadata": {
                    "month_simulation_time_s": sim_time_s,
                    "interconnect_mw": oa.interconnect_mw,
                    "dt": oa.dt,
                    "n_rows": len(md),
                },
                "components": {},
                "categories": {},
                "plant": {},
                "market_metrics": {},
            }

            total_energy = 0.0
            total_rev_rt = 0.0
            total_rev_da = 0.0

            for comp in oa.components:
                n = comp.name
                energy = md[f"{n}_energy_mwh"].sum()
                rev_rt = md[f"{n}_revenue_rt"].sum()
                rev_da = md[f"{n}_revenue_da"].sum()

                total_energy += energy
                total_rev_rt += rev_rt
                total_rev_da += rev_da

                cm: dict = {
                    "component_type": comp.component_type,
                    "category": comp.category,
                    "energy_mwh": energy,
                    "revenue_rt_k": rev_rt / 1e3,
                    "revenue_da_k": rev_da / 1e3,
                    "revenue_rt_if_all_paid_k": (
                        md[f"{n}_revenue_rt_if_all_paid"].sum() / 1e3
                    ),
                    "revenue_da_if_all_paid_k": (
                        md[f"{n}_revenue_da_if_all_paid"].sum() / 1e3
                    ),
                }

                if comp.category == "generator":
                    cm["revenue_rt_excess_loss_k"] = (
                        md[f"{n}_revenue_rt_excess_loss"].sum() / 1e3
                    )
                    cm["revenue_da_excess_loss_k"] = (
                        md[f"{n}_revenue_da_excess_loss"].sum() / 1e3
                    )

                if comp.category == "storage":
                    cm["energy_discharge_mwh"] = md[md[f"{n}_energy_mwh"] > 0][
                        f"{n}_energy_mwh"
                    ].sum()
                    cm["energy_charge_mwh"] = md[md[f"{n}_energy_mwh"] < 0][
                        f"{n}_energy_mwh"
                    ].sum()
                    cm["revenue_rt_discharge_k"] = (
                        md[f"{n}_revenue_rt_discharge"].sum() / 1e3
                    )
                    cm["revenue_rt_charge_k"] = md[f"{n}_revenue_rt_charge"].sum() / 1e3
                    cm["revenue_rt_charge_if_all_paid_k"] = (
                        md[f"{n}_revenue_rt_charge_if_all_paid"].sum() / 1e3
                    )
                    cm["revenue_rt_charge_savings_k"] = (
                        md[f"{n}_revenue_rt_charge_savings"].sum() / 1e3
                    )
                    cm["revenue_da_charge_k"] = md[f"{n}_revenue_da_charge"].sum() / 1e3
                    cm["revenue_da_charge_if_all_paid_k"] = (
                        md[f"{n}_revenue_da_charge_if_all_paid"].sum() / 1e3
                    )
                    cm["revenue_da_charge_savings_k"] = (
                        md[f"{n}_revenue_da_charge_savings"].sum() / 1e3
                    )
                    cm.update(self._compute_mileage_metrics(md, oa, n))

                mm["components"][n] = cm

            for category, names in [
                ("generator", oa.generators),
                ("storage", oa.storage),
                ("load", oa.loads),
            ]:
                if not names:
                    continue
                mm["categories"][category] = {
                    "energy_mwh": sum(mm["components"][n]["energy_mwh"] for n in names),
                    "revenue_rt_k": sum(
                        mm["components"][n]["revenue_rt_k"] for n in names
                    ),
                    "revenue_da_k": sum(
                        mm["components"][n]["revenue_da_k"] for n in names
                    ),
                }

            cf = (
                total_energy / (oa.interconnect_mw * sim_hours)
                if oa.interconnect_mw * sim_hours > 0
                else float("nan")
            )
            mm["plant"]["total_energy_mwh"] = total_energy
            mm["plant"]["capacity_factor"] = cf
            mm["plant"]["total_revenue_rt_k"] = total_rev_rt / 1e3
            mm["plant"]["total_revenue_da_k"] = total_rev_da / 1e3
            mm["plant"].update(self._surplus_metrics(md))

            avg_rt = md["lmp_rt"].mean()
            avg_rev = avg_rt * total_energy
            mm["market_metrics"]["avg_price_rt"] = avg_rt
            mm["market_metrics"]["avg_price_revenue_k"] = avg_rev / 1e3
            mm["market_metrics"]["value_factor"] = (
                total_rev_rt / avg_rev
                if total_rev_rt > 0 and avg_rev > 0
                else float("nan")
            )

            monthly_metrics[month_str] = mm

        return monthly_metrics

    def _display_monthly_summary(self, monthly_metrics: dict):
        """Display monthly metrics summary table.

        Args:
            monthly_metrics (dict): Monthly metrics dictionary.
        """
        months = sorted(monthly_metrics.keys())
        if not months:
            print("No monthly data available.")
            return

        row_defs = [
            ("PlantEnergy", "plant", "total_energy_mwh", ",.0f", "MWh"),
            ("PlantCF", "plant", "capacity_factor", ".1%", "%"),
            ("RevenueRT", "plant", "total_revenue_rt_k", ",.0f", "K$"),
            ("RevenueDA", "plant", "total_revenue_da_k", ",.0f", "K$"),
            ("AvgPriceRT", "market_metrics", "avg_price_rt", ".1f", "$/MWh"),
            ("ValueFactor", "market_metrics", "value_factor", ".3f", "-"),
        ]

        print("\n" + "=" * 120)
        print("MONTHLY SUMMARY TABLE")
        print("=" * 120)

        month_w, col_w = 10, 12
        header = f"{'Month':<{month_w}}"
        for label, *_ in row_defs:
            header += f"{label:>{col_w}}"
        print(header)

        units_row = f"{'':<{month_w}}"
        for _, _, _, _, unit in row_defs:
            units_row += f"({unit})".rjust(col_w)
        print(units_row)
        print("-" * len(header))

        for month in months:
            md = monthly_metrics[month]
            row = f"{month:<{month_w}}"
            for _, section, key, fmt, _ in row_defs:
                try:
                    val = md[section][key]
                    row += f"{val:{fmt}}".rjust(col_w)
                except (KeyError, TypeError, ValueError):
                    row += "N/A".rjust(col_w)
            print(row)

        print("=" * 120)

    # ------------------------------------------------------------------
    # Scenario comparison
    # ------------------------------------------------------------------

    def compare_scenarios(
        self,
        display_format: str = "table",
        output_csv: str | None = None,
        *,
        include_pct_change: bool = False,
    ) -> pd.DataFrame:
        """Compare metrics across scenarios.

        Args:
            display_format (str, optional): 'table' or 'raw'. Defaults to 'table'.
            output_csv (str, optional): Path to save CSV. Defaults to None.
            include_pct_change (bool, optional): If True, append a column with
                percent change from the first scenario to the second (exactly two
                scenarios required). Values look like '+24.15%' (two decimals,
                signed). Defaults to False.

        Returns:
            pd.DataFrame: Comparison table.

        Raises:
            ValueError: If not multi-scenario, metrics not computed, or
                ``include_pct_change`` is True but the number of scenarios is not 2.
        """
        if not self.is_multi_scenario:
            raise ValueError("Comparison requires multiple OutputAnalysis objects")
        if self.metrics is None:
            raise ValueError("Must call compute_metrics() before comparing scenarios")
        if include_pct_change and len(self.scenario_names) != 2:
            raise ValueError(
                "include_pct_change requires exactly two scenarios; "
                f"got {len(self.scenario_names)}."
            )

        metric_defs = self._build_comparison_definitions()

        comparison_data: dict[str, list] = {n: [] for n in self.scenario_names}
        for _label, path in metric_defs:
            if path == "":
                for s in self.scenario_names:
                    comparison_data[s].append("")
            else:
                for sname, m in zip(self.scenario_names, self.metrics, strict=False):
                    comparison_data[sname].append(self._extract_path(m, path))

        labels = [label for label, _ in metric_defs]
        df_cmp = pd.DataFrame(comparison_data, index=labels)

        if include_pct_change:
            metrics_list = self.metrics
            pct_series = []
            for _label, path in metric_defs:
                if path == "":
                    pct_series.append("")
                else:
                    v0 = self._extract_path_numeric(metrics_list[0], path)
                    v1 = self._extract_path_numeric(metrics_list[1], path)
                    pct_series.append(self._format_pct_change(v0, v1))
            df_cmp[self._PCT_CHANGE_COLUMN] = pct_series

        if display_format == "table":
            self._print_comparison_table(df_cmp)

        if output_csv is not None:
            out = Path(output_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            df_cmp.to_csv(out)
            print(f"\nComparison table saved to: {output_csv}")

        return df_cmp

    def _build_comparison_definitions(self) -> list[tuple[str, str]]:
        """Build metric definitions for scenario comparison.

        Returns:
            list[tuple[str, str]]: (display_label, dot-path) pairs.
        """
        sample = self.metrics[0] if self.is_multi_scenario else self.metrics
        defs: list[tuple[str, str]] = [("Plant Metrics", "")]
        defs.append(("  Plant Energy (MWh)", "plant.total_energy_mwh"))
        defs.append(("  Plant CF", "plant.capacity_factor"))
        defs.append(("  Total Revenue RT (K$)", "plant.total_revenue_rt_k"))
        defs.append(("  Total Revenue DA (K$)", "plant.total_revenue_da_k"))

        if sample.get("components"):
            defs.append(("Component Metrics", ""))
            for name, _cm in sample["components"].items():
                defs.append((f"  {name} energy (MWh)", f"components.{name}.energy_mwh"))
                defs.append(
                    (f"  {name} revenue RT (K$)", f"components.{name}.revenue_rt_k")
                )

        defs.append(("Market Metrics", ""))
        defs.append(("  Avg Price RT ($/MWh)", "market_metrics.avg_price_rt"))
        defs.append(("  Value Factor", "market_metrics.value_factor"))
        return defs

    @staticmethod
    def _extract_path(metrics: dict, path: str) -> str:
        """Navigate a dotted path into a nested dict.

        Args:
            metrics (dict): Root metrics dict.
            path (str): Dot-separated key path.

        Returns:
            str: Formatted value string.
        """
        try:
            val = metrics
            for key in path.split("."):
                val = val[key]
            if isinstance(val, float):
                if "capacity_factor" in path or "value_factor" in path:
                    return f"{val:.3f}"
                return f"{val:,.1f}"
            return str(val)
        except (KeyError, TypeError):
            return "N/A"

    @staticmethod
    def _extract_path_numeric(metrics: dict, path: str) -> float | None:
        """Read a numeric metric from a dotted path.

        Args:
            metrics (dict): Root metrics dict.
            path (str): Dot-separated key path.

        Returns:
            float | None: Parsed value, or None if missing or non-numeric.
        """
        try:
            val = metrics
            for key in path.split("."):
                val = val[key]
            return float(val)
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _format_pct_change(first: float | None, second: float | None) -> str:
        """Format relative percent change from first to second value.

        Args:
            first (float | None): Baseline (first scenario) value.
            second (float | None): Comparison (second scenario) value.

        Returns:
            str: Signed percentage like '+24.15%', or 'N/A' when undefined.
        """
        if first is None or second is None:
            return "N/A"
        if first == 0:
            if second == 0:
                return "+0.00%"
            return "N/A"
        pct = (second - first) / first * 100.0
        return f"{pct:+.2f}%"

    def _print_comparison_table(self, df_cmp: pd.DataFrame):
        """Print a formatted comparison table.

        Args:
            df_cmp (pd.DataFrame): Comparison dataframe.
        """
        col_w = max(12, max(len(n) for n in self.scenario_names) + 2)
        has_pct = self._PCT_CHANGE_COLUMN in df_cmp.columns
        pct_col_w = max(12, len(self._PCT_CHANGE_COLUMN) + 2) if has_pct else 0
        total_w = 35 + col_w * len(self.scenario_names) + pct_col_w

        print("\n" + "=" * total_w)
        print("SCENARIO COMPARISON")
        print("=" * total_w)

        header = f"{'Metric':<35}"
        for s in self.scenario_names:
            header += f"{s:>{col_w}}"
        if has_pct:
            header += f"{self._PCT_CHANGE_COLUMN:>{pct_col_w}}"
        print(header)
        print("-" * total_w)

        for label in df_cmp.index:
            vals = [df_cmp.loc[label, s] for s in self.scenario_names]
            if all(v == "" for v in vals):
                print(f"\n{label}")
                print("-" * len(label))
            elif all(v == "N/A" for v in vals):
                continue
            else:
                row = f"{label:<35}"
                for v in vals:
                    try:
                        fv = float(v)
                        if fv == 0:
                            rv = "0.0000"
                        else:
                            dp = max(
                                0,
                                -int(math.floor(math.log10(abs(fv)))) + 3,
                            )
                            rv = f"{fv:.{dp}f}"
                    except (ValueError, TypeError):
                        rv = v
                    row += f"{rv:>{col_w}}"
                if has_pct:
                    pct_val = df_cmp.loc[label, self._PCT_CHANGE_COLUMN]
                    row += f"{pct_val:>{pct_col_w}}"
                print(row)

        print("=" * total_w)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_metrics(self, output_file: str, monthly: bool = False):
        """Save computed metrics to file.

        Args:
            output_file (str): Path (pickle or JSON based on extension).
            monthly (bool, optional): Save monthly metrics. Defaults to False.
        """
        data = self.monthly_metrics if monthly else self.metrics
        if data is None:
            raise ValueError(
                "Must call compute_metrics() or compute_monthly_metrics() first"
            )

        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)

        if out.suffix.lower() == ".json":
            with open(out, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"Metrics saved to JSON file: {output_file}")
        else:
            with open(out, "wb") as f:
                pickle.dump(data, f)
            print(f"Metrics saved to pickle file: {output_file}")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def available_metrics(self) -> list[str]:
        """Return all available metric key paths.

        Returns:
            list[str]: Sorted list of dot-separated metric paths.
        """
        if self.metrics is None:
            raise ValueError("Must call compute_metrics() first")
        sample = self.metrics[0] if self.is_multi_scenario else self.metrics
        paths: list[str] = []
        self._collect_paths(sample, "", paths)
        paths.sort()
        print("Available metrics:")
        for p in paths:
            print(f"  {p}")
        return paths

    @staticmethod
    def _collect_paths(d: dict, prefix: str, out: list[str]):
        """Recursively collect dot-separated key paths from a nested dict.

        Args:
            d (dict): Input dict.
            prefix (str): Current prefix.
            out (list[str]): Output list.
        """
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                TotalMetrics._collect_paths(v, path, out)
            else:
                out.append(path)

    def _get_metric_value(self, metrics: dict, metric_name: str) -> float:
        """Extract a metric value by dot-path.

        Args:
            metrics (dict): Metrics dictionary.
            metric_name (str): Dot-separated path.

        Returns:
            float: The metric value.
        """
        val = metrics
        for key in metric_name.split("."):
            val = val[key]
        return val

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_compare_scenarios(
        self,
        metrics: str | list[str],
        plot_type: str = "bar",
        **kwargs,
    ):
        """Plot metric(s) comparing scenarios.

        Args:
            metrics (str | list[str]): Dot-path metric name(s).
            plot_type (str, optional): 'line', 'scatter', or 'bar'.
                Defaults to 'bar'.
            **kwargs: Passed to the plot function.

        Returns:
            tuple: (fig, ax) or (fig, axarr).
        """
        if not self.is_multi_scenario:
            raise ValueError("Comparison requires multiple scenarios")
        if self.metrics is None:
            raise ValueError("Must call compute_metrics() before plotting")

        if isinstance(metrics, str):
            metrics = [metrics]

        x = range(len(self.scenario_names))

        if len(metrics) == 1:
            fig, ax = plt.subplots()
            values = [self._get_metric_value(m, metrics[0]) for m in self.metrics]
            self._plot_single(ax, x, values, plot_type, **kwargs)
            ax.set_ylabel(metrics[0])
            ax.set_xticks(list(x))
            ax.set_xticklabels(self.scenario_names, rotation=45, ha="right")
            ax.grid(True)
            fig.tight_layout()
            return fig, ax
        else:
            fig, axarr = plt.subplots(len(metrics), 1, sharex=True)
            for i, metric in enumerate(metrics):
                values = [self._get_metric_value(m, metric) for m in self.metrics]
                self._plot_single(axarr[i], x, values, plot_type, **kwargs)
                axarr[i].set_ylabel(metric.split(".")[-1])
                axarr[i].grid(True)
            axarr[-1].set_xticks(list(x))
            axarr[-1].set_xticklabels(self.scenario_names, rotation=45, ha="right")
            fig.tight_layout()
            return fig, axarr

    @staticmethod
    def _plot_single(ax, x, values, plot_type: str, **kwargs):
        """Plot values on an axis.

        Args:
            ax: Matplotlib axis.
            x: X values.
            values: Y values.
            plot_type (str): Plot type.
            **kwargs: Additional keyword arguments.
        """
        if plot_type == "line":
            ax.plot(x, values, **kwargs)
        elif plot_type == "scatter":
            ax.scatter(x, values, **kwargs)
        elif plot_type == "bar":
            ax.bar(x, values, **kwargs)
        else:
            raise ValueError(
                f"Invalid plot_type: {plot_type}. Use 'line', 'scatter', or 'bar'."
            )

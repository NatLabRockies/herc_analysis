"""OutputAnalysis class for processing Hercules simulation output."""

import numpy as np
import pandas as pd
from hercules.hercules_output import HerculesOutput

from herc_analysis.constants import (
    COMPONENT_TYPE_TO_CATEGORY,
    ComponentInfo,
)


class OutputAnalysis:
    """Process and analyze hybrid plant simulation data from a Hercules H5 file.

    All components are handled generically using their component_name,
    component_type, and component_category.  Derived columns use the
    component_name as prefix so that multiple instances of the same type
    coexist naturally.
    """

    def __init__(
        self,
        hercules_output_file="outputs/hercules_output.h5",
    ):
        """Initialize OutputAnalysis from a Hercules H5 output file.

        Args:
            hercules_output_file (str, optional): Path to the Hercules
                simulation output H5 file.
                Defaults to "outputs/hercules_output.h5".
        """
        print("Reading Hercules output H5 file...")
        ho = HerculesOutput(hercules_output_file)
        df = ho.df.copy()

        self.ho = ho
        self.h_dict = ho.h_dict
        self.dt_log = float(ho.dt_log)
        self.dt = self.dt_log
        self.dt_sim = float(ho.dt_sim)
        self.starttime = float(ho.starttime)
        self.endtime = float(ho.endtime)
        self.total_simulation_time_s = float(ho.total_simulation_time)
        self.n_rows = int(ho.total_rows_written)
        self.total_time_wall = float(ho.total_time_wall)

        self.columns = df.columns
        self.num_columns = df.shape[1]

        self.interconnect_mw = (
            float(self.h_dict["plant"]["interconnect_limit"]) / 1000.0
        )

        self._discover_components()

        df = self._process_external_signals(df)

        for comp in self.components:
            df = self._process_component(df, comp)

        df = self._finalize_columns(df)
        self.df = self._compute_plant_level_metrics(df)

        print("Data loaded into pandas dataframe for analysis...")
        self.ho.print_metadata()

    # ------------------------------------------------------------------
    # Component discovery
    # ------------------------------------------------------------------

    def _discover_components(self):
        """Discover plant components from h_dict using component_type.

        Populates ``self.components`` with :class:`ComponentInfo` objects and
        convenience lists ``self.generators``, ``self.storage``, ``self.loads``.
        """
        self.components: list[ComponentInfo] = []

        for key, val in self.h_dict.items():
            if not isinstance(val, dict):
                continue

            component_type = val.get("component_type", "")
            if not component_type:
                continue

            category = COMPONENT_TYPE_TO_CATEGORY.get(component_type)
            if category is None:
                print(
                    f"Warning: unknown component_type '{component_type}' "
                    f"for component '{key}' -- skipping"
                )
                continue

            self.components.append(ComponentInfo(key, component_type, category))

        self.generators = [c.name for c in self.components if c.category == "generator"]
        self.storage = [c.name for c in self.components if c.category == "storage"]
        self.loads = [c.name for c in self.components if c.category == "load"]

        print(
            f"Discovered {len(self.components)} components: "
            f"{len(self.generators)} generators, "
            f"{len(self.storage)} storage, "
            f"{len(self.loads)} loads"
        )

    # ------------------------------------------------------------------
    # External signals (LMP)
    # ------------------------------------------------------------------

    def _process_external_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process external signals including LMP data.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with LMP columns added.
        """
        print("Processing component-independent data...")

        df["time"] = df["time"].astype(float)

        df_unique_times = df["time"].nunique()
        print(f"DF unique time values: {df_unique_times} (total rows: {df.shape[0]})")

        if df_unique_times != df.shape[0]:
            print("Warning: df has duplicate time values - removing duplicates")
            df = df.drop_duplicates(subset=["time"], keep="first")
            print(f"After removing duplicates, df has {df.shape[0]} rows")

        if "external_signals" in self.h_dict and self.h_dict["external_signals"]:
            print("LMP data found in external_signals section of H5 file...")
            lmp_rt_col = None
            lmp_da_col = None

            for col in df.columns:
                if "lmp_rt" in col.lower():
                    lmp_rt_col = col
                elif "lmp_da" in col.lower():
                    lmp_da_col = col

            if lmp_rt_col and lmp_da_col:
                df["lmp_rt"] = df[lmp_rt_col]
                df["lmp_da"] = df[lmp_da_col]
                print(
                    f"LMP data extracted from external_signals: "
                    f"{lmp_rt_col}, {lmp_da_col}"
                )
            else:
                print("Warning: LMP data not found in external_signals columns.")
                df["lmp_rt"] = 0.0
                df["lmp_da"] = 0.0
        else:
            print("No external signals (LMP data) found in H5 file.")
            df["lmp_rt"] = 0.0
            df["lmp_da"] = 0.0

        if "time_utc" in df.columns:
            df["lmp_rt_hourly"] = df.groupby(
                [df["time_utc"].dt.date, df["time_utc"].dt.hour]
            )["lmp_rt"].transform("mean")
        else:
            df["lmp_rt_hourly"] = df.groupby(df["time"] // 3600)["lmp_rt"].transform(
                "mean"
            )

        # Plant-level locally generated power (kW -> MW). Used to detect when
        # local generation exceeds the interconnect so storage charging from
        # that excess can be priced at $0/MWh instead of LMP.
        if "plant.locally_generated_power" in df.columns:
            df["plant_locally_generated_power_mw"] = (
                df["plant.locally_generated_power"] / 1000.0
            )
        else:
            print(
                "Warning: 'plant.locally_generated_power' not found in H5; "
                "skipping excess-charging cost correction. Storage charging "
                "will be priced at LMP for all energy, which may overstate "
                "charging cost when local generation exceeds the "
                "interconnect."
            )

        return df

    # ------------------------------------------------------------------
    # Generic component processing
    # ------------------------------------------------------------------

    def _process_component(self, df: pd.DataFrame, comp: ComponentInfo) -> pd.DataFrame:
        """Process a single component: derive power, energy, revenue columns.

        For storage components, also computes charge/discharge split and SOC.

        Args:
            df (pd.DataFrame): Input dataframe.
            comp (ComponentInfo): Component metadata.

        Returns:
            pd.DataFrame: Dataframe with derived component columns added.
        """
        name = comp.name
        print(
            f"Processing component '{name}' "
            f"(type={comp.component_type}, category={comp.category})..."
        )

        power_col = f"{name}.power"
        if power_col in df.columns:
            df[f"{name}_power_mw"] = df[power_col] / 1000.0
        else:
            print(f"  Warning: {power_col} not found -- using zeros")
            df[f"{name}_power_mw"] = 0.0

        setpoint_col = f"{name}.power_setpoint"
        if setpoint_col in df.columns:
            df[f"{name}_power_setpoint_mw"] = df[setpoint_col] / 1000.0

        df[f"{name}_energy_mwh"] = df[f"{name}_power_mw"] * self.dt / 3600
        df[f"{name}_revenue_rt"] = df["lmp_rt"] * df[f"{name}_energy_mwh"]
        df[f"{name}_revenue_da"] = df["lmp_da"] * df[f"{name}_energy_mwh"]

        # Sanity baseline for every component: revenue as if every MWh were
        # priced at LMP. For generators, the excess-generation-revenue
        # correction below will reduce {name}_revenue_rt/_da to reflect that
        # generation above the interconnect is not delivered to the grid.
        # The baselines are deliberately *not* modified by any correction.
        df[f"{name}_revenue_rt_if_all_paid"] = df[f"{name}_revenue_rt"]
        df[f"{name}_revenue_da_if_all_paid"] = df[f"{name}_revenue_da"]

        if comp.category == "generator":
            # Diagnostics populated by
            # _apply_excess_generation_revenue_correction.
            df[f"{name}_excess_local_generation_mw"] = 0.0
            df[f"{name}_revenue_rt_excess_loss"] = 0.0
            df[f"{name}_revenue_da_excess_loss"] = 0.0

        if comp.category == "storage":
            df[f"{name}_revenue_rt_discharge"] = np.where(
                df[f"{name}_energy_mwh"] > 0, df[f"{name}_revenue_rt"], 0
            )
            df[f"{name}_revenue_rt_charge"] = np.where(
                df[f"{name}_energy_mwh"] < 0, df[f"{name}_revenue_rt"], 0
            )
            df[f"{name}_revenue_da_discharge"] = np.where(
                df[f"{name}_energy_mwh"] > 0, df[f"{name}_revenue_da"], 0
            )
            df[f"{name}_revenue_da_charge"] = np.where(
                df[f"{name}_energy_mwh"] < 0, df[f"{name}_revenue_da"], 0
            )

            # Sanity baseline: charge cost as if every MWh were paid at LMP.
            # These columns are never modified by the excess-charge correction
            # so users can compare 'paid' vs 'if all paid' to see the value of
            # internal (excess-absorbed) charging.
            df[f"{name}_revenue_rt_charge_if_all_paid"] = df[
                f"{name}_revenue_rt_charge"
            ]
            df[f"{name}_revenue_da_charge_if_all_paid"] = df[
                f"{name}_revenue_da_charge"
            ]

            # Diagnostics populated by _apply_excess_charge_correction.
            df[f"{name}_excess_absorbed_mw"] = 0.0
            df[f"{name}_revenue_rt_charge_savings"] = 0.0
            df[f"{name}_revenue_da_charge_savings"] = 0.0

            soc_col = f"{name}.soc"
            if soc_col in df.columns:
                df[f"{name}_soc"] = df[soc_col]
            else:
                print(f"  Note: {soc_col} not logged (optional)")
                df[f"{name}_soc"] = 0.0

        return df

    # ------------------------------------------------------------------
    # Column selection
    # ------------------------------------------------------------------

    def _finalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select relevant columns, keeping base, per-component, and raw signal columns.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with only selected columns.
        """
        print("Finalizing data processing...")

        keep: list[str] = ["time", "lmp_rt", "lmp_da", "lmp_rt_hourly"]
        if "time_utc" in df.columns:
            keep.append("time_utc")
        if "plant_locally_generated_power_mw" in df.columns:
            keep.append("plant_locally_generated_power_mw")

        for comp in self.components:
            name = comp.name
            derived = [
                f"{name}_power_mw",
                f"{name}_power_setpoint_mw",
                f"{name}_energy_mwh",
                f"{name}_revenue_rt",
                f"{name}_revenue_da",
                f"{name}_revenue_rt_if_all_paid",
                f"{name}_revenue_da_if_all_paid",
            ]
            if comp.category == "generator":
                derived += [
                    f"{name}_excess_local_generation_mw",
                    f"{name}_revenue_rt_excess_loss",
                    f"{name}_revenue_da_excess_loss",
                ]
            if comp.category == "storage":
                derived += [
                    f"{name}_revenue_rt_discharge",
                    f"{name}_revenue_rt_charge",
                    f"{name}_revenue_da_discharge",
                    f"{name}_revenue_da_charge",
                    f"{name}_revenue_rt_charge_if_all_paid",
                    f"{name}_revenue_da_charge_if_all_paid",
                    f"{name}_revenue_rt_charge_savings",
                    f"{name}_revenue_da_charge_savings",
                    f"{name}_excess_absorbed_mw",
                    f"{name}_soc",
                ]
            keep.extend(c for c in derived if c in df.columns)

            raw_prefix = f"{name}."
            keep.extend(
                c for c in df.columns if c.startswith(raw_prefix) and c not in keep
            )

        external = [c for c in df.columns if "external_signals" in c]
        keep.extend(c for c in external if c not in keep)

        return df[[c for c in keep if c in df.columns]]

    # ------------------------------------------------------------------
    # Category and plant aggregates
    # ------------------------------------------------------------------

    def _apply_excess_charge_correction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zero out charging cost for storage absorbing locally-generated excess.

        When ``plant_locally_generated_power_mw`` exceeds ``interconnect_mw``,
        the excess would be curtailed if not absorbed by storage. Any portion
        of storage charging that absorbs this excess should be priced at
        $0/MWh (RT and DA) rather than at the prevailing LMP.

        For each row, the excess is allocated across charging storage
        components proportionally to their charging power. Per-component
        ``_revenue_rt``, ``_revenue_da``, ``_revenue_rt_charge``, and
        ``_revenue_da_charge`` columns are reduced (charging cost moves
        toward zero). The ``_revenue_*_charge_if_all_paid`` baselines created
        in ``_process_component`` are deliberately *not* modified, so users
        can compare paid vs. as-if-all-paid charging cost.

        The plant-level diagnostic ``excess_local_generation_mw`` and
        per-storage ``{name}_excess_absorbed_mw`` and
        ``{name}_revenue_*_charge_savings`` columns are populated here.

        No-ops cleanly when ``plant_locally_generated_power_mw`` is missing,
        no storage components exist, or no excess is observed.

        Args:
            df (pd.DataFrame): Dataframe with per-component power and revenue
                columns already computed.

        Returns:
            pd.DataFrame: Dataframe with corrected storage revenue columns
                and excess diagnostics.
        """
        df["excess_local_generation_mw"] = 0.0

        if "plant_locally_generated_power_mw" not in df.columns:
            return df
        if not self.storage:
            return df

        excess = (df["plant_locally_generated_power_mw"] - self.interconnect_mw).clip(
            lower=0
        )
        df["excess_local_generation_mw"] = excess

        if not (excess > 0).any():
            return df

        # Per-storage charging power (MW, positive when charging).
        charge_mw_by_name = {
            name: (-df[f"{name}_power_mw"]).clip(lower=0) for name in self.storage
        }
        total_charge_mw = sum(charge_mw_by_name.values())

        # Allocation scale per row: fraction of each storage's charging that
        # is absorbed from excess local generation. With proportional sharing,
        # each component absorbs charge_mw_i * scale, where
        # scale = min(1, excess / total_charge).
        scale = (
            (excess / total_charge_mw.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
        )

        energy_per_mw = self.dt / 3600.0

        for name in self.storage:
            absorbed_mw = charge_mw_by_name[name] * scale
            savings_rt = df["lmp_rt"] * absorbed_mw * energy_per_mw
            savings_da = df["lmp_da"] * absorbed_mw * energy_per_mw

            df[f"{name}_excess_absorbed_mw"] = absorbed_mw
            df[f"{name}_revenue_rt_charge_savings"] = savings_rt
            df[f"{name}_revenue_da_charge_savings"] = savings_da

            # Reduce charging cost (move toward zero) on both the
            # split charge column and the total component revenue column.
            df[f"{name}_revenue_rt_charge"] = (
                df[f"{name}_revenue_rt_charge"] + savings_rt
            )
            df[f"{name}_revenue_da_charge"] = (
                df[f"{name}_revenue_da_charge"] + savings_da
            )
            df[f"{name}_revenue_rt"] = df[f"{name}_revenue_rt"] + savings_rt
            df[f"{name}_revenue_da"] = df[f"{name}_revenue_da"] + savings_da

        return df

    def _apply_excess_generation_revenue_correction(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Reduce generator revenue for any locally-generated power above the interconnect.

        When ``plant_locally_generated_power_mw`` exceeds ``interconnect_mw``,
        the excess is not delivered to the grid -- it is either curtailed or
        absorbed by storage. Either way, generators should not be paid for
        the excess MWh. The plant-level ``excess_local_generation_mw`` is
        allocated proportionally across generators by their per-row power
        share, mirroring the storage charging fix.

        Per-generator ``_revenue_rt`` and ``_revenue_da`` are reduced in
        place; the ``_revenue_*_if_all_paid`` baselines from
        ``_process_component`` are deliberately left untouched.

        No-ops cleanly when ``excess_local_generation_mw`` is missing/zero
        or when there are no generators.

        Args:
            df (pd.DataFrame): Dataframe with per-component power and revenue
                columns plus ``excess_local_generation_mw``.

        Returns:
            pd.DataFrame: Dataframe with corrected generator revenue columns
                and per-generator excess diagnostics.
        """
        if "excess_local_generation_mw" not in df.columns:
            return df
        if not self.generators:
            return df

        excess = df["excess_local_generation_mw"]
        if not (excess > 0).any():
            return df

        # Per-generator power (clip negatives so a momentarily-negative
        # generator can't flip the proportional allocation).
        gen_mw_by_name = {
            name: df[f"{name}_power_mw"].clip(lower=0) for name in self.generators
        }
        total_gen_mw = sum(gen_mw_by_name.values())

        # Allocation share per row: gen_i / total_gen.
        share = (1.0 / total_gen_mw.replace(0, np.nan)).fillna(0.0)
        energy_per_mw = self.dt / 3600.0

        for name in self.generators:
            gen_excess_mw = gen_mw_by_name[name] * excess * share
            loss_rt = df["lmp_rt"] * gen_excess_mw * energy_per_mw
            loss_da = df["lmp_da"] * gen_excess_mw * energy_per_mw

            df[f"{name}_excess_local_generation_mw"] = gen_excess_mw
            df[f"{name}_revenue_rt_excess_loss"] = loss_rt
            df[f"{name}_revenue_da_excess_loss"] = loss_da

            df[f"{name}_revenue_rt"] = df[f"{name}_revenue_rt"] - loss_rt
            df[f"{name}_revenue_da"] = df[f"{name}_revenue_da"] - loss_da

        return df

    def _compute_category_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sum per-component power/energy/revenue into per-category totals.

        Args:
            df (pd.DataFrame): Input dataframe with per-component columns.

        Returns:
            pd.DataFrame: Dataframe with category aggregate columns added.
        """
        for category, names in [
            ("generator", self.generators),
            ("storage", self.storage),
            ("load", self.loads),
        ]:
            if not names:
                continue
            for suffix in ("_power_mw", "_energy_mwh", "_revenue_rt", "_revenue_da"):
                cols = [f"{n}{suffix}" for n in names if f"{n}{suffix}" in df.columns]
                if cols:
                    df[f"{category}_total{suffix}"] = df[cols].sum(axis=1)

        return df

    def _compute_plant_level_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute plant-level derived metrics.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with plant-level metrics added.
        """
        print("Computing derived columns...")

        power_cols = [
            f"{c.name}_power_mw" for c in self.components if f"{c.name}_power_mw" in df
        ]
        df["total_plant_power_mw"] = df[power_cols].sum(axis=1) if power_cols else 0.0
        df["total_plant_energy_mwh"] = df["total_plant_power_mw"] * self.dt / 3600

        # Apply the excess-local-generation charging cost correction before
        # computing category aggregates and plant-level revenue, so all sums
        # reflect the corrected per-component revenue columns.
        df = self._apply_excess_charge_correction(df)
        df = self._apply_excess_generation_revenue_correction(df)

        df = self._compute_category_aggregates(df)

        # Plant-level revenue is the sum of per-component revenue. This stays
        # consistent with the excess-charge correction above (otherwise it
        # would double-count the LMP-priced charging energy that was waived).
        rev_rt_cols = [
            f"{c.name}_revenue_rt"
            for c in self.components
            if f"{c.name}_revenue_rt" in df.columns
        ]
        rev_da_cols = [
            f"{c.name}_revenue_da"
            for c in self.components
            if f"{c.name}_revenue_da" in df.columns
        ]
        df["total_plant_revenue_rt"] = (
            df[rev_rt_cols].sum(axis=1) if rev_rt_cols else 0.0
        )
        df["total_plant_revenue_da"] = (
            df[rev_da_cols].sum(axis=1) if rev_da_cols else 0.0
        )

        df["surplus_capacity_mw"] = self.interconnect_mw - df["total_plant_power_mw"]
        df["surplus_capacity_energy_mwh"] = df["surplus_capacity_mw"] * self.dt / 3600

        df["surplus_capacity_revenue_rt"] = (
            df["surplus_capacity_energy_mwh"] * df["lmp_rt"]
        )
        df["surplus_capacity_revenue_da"] = (
            df["surplus_capacity_energy_mwh"] * df["lmp_da"]
        )

        lmp_rt_pos = np.where(df["lmp_rt"] > 0, df["lmp_rt"], 0)
        lmp_da_pos = np.where(df["lmp_da"] > 0, df["lmp_da"], 0)
        lmp_rt_neg = np.where(df["lmp_rt"] < 0, df["lmp_rt"], 0)
        lmp_da_neg = np.where(df["lmp_da"] < 0, df["lmp_da"], 0)

        df["surplus_capacity_revenue_rt_positive_lmp"] = (
            df["surplus_capacity_energy_mwh"] * lmp_rt_pos
        )
        df["surplus_capacity_revenue_da_positive_lmp"] = (
            df["surplus_capacity_energy_mwh"] * lmp_da_pos
        )
        df["surplus_capacity_revenue_rt_negative_lmp"] = (
            -1 * df["surplus_capacity_energy_mwh"] * lmp_rt_neg
        )
        df["surplus_capacity_revenue_da_negative_lmp"] = (
            -1 * df["surplus_capacity_energy_mwh"] * lmp_da_neg
        )

        df["ideal_capacity_revenue_rt_positive_lmp"] = (
            self.interconnect_mw * lmp_rt_pos * self.dt / 3600
        )
        df["ideal_capacity_revenue_da_positive_lmp"] = (
            self.interconnect_mw * lmp_da_pos * self.dt / 3600
        )
        df["ideal_capacity_revenue_rt_negative_lmp"] = (
            -1 * self.interconnect_mw * lmp_rt_neg * self.dt / 3600
        )
        df["ideal_capacity_revenue_da_negative_lmp"] = (
            -1 * self.interconnect_mw * lmp_da_neg * self.dt / 3600
        )

        if df["total_plant_power_mw"].max() > self.interconnect_mw:
            print("Warning: Total plant power exceeds interconnect capacity.")

        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_time_axis_info(self):
        """Determine the appropriate time column and axis label for plotting.

        Returns:
            tuple: (time_column_name, x_axis_label)
        """
        if "time_utc" in self.df.columns and self.df["time_utc"].notna().any():
            return "time_utc", "Time (UTC)"
        return "time", "Time (seconds)"

    def get_component(self, name: str) -> ComponentInfo | None:
        """Look up a ComponentInfo by name.

        Args:
            name (str): Component name.

        Returns:
            ComponentInfo | None: The matching ComponentInfo, or None.
        """
        for c in self.components:
            if c.name == name:
                return c
        return None

    def get_signal_columns(self, name: str, signal: str) -> list[str]:
        """Return dataframe columns matching ``{name}.{signal}.*``.

        Useful for array signals like turbine_powers.

        Args:
            name (str): Component name.
            signal (str): Signal name (e.g. 'turbine_powers').

        Returns:
            list[str]: Matching column names, sorted.
        """
        prefix = f"{name}.{signal}."
        exact = f"{name}.{signal}"
        return sorted(c for c in self.df.columns if c.startswith(prefix) and c != exact)

"""Cross-scenario metric comparison for Hercules analysis.

This module provides :class:`ScenarioComparison`, a generic engine for
comparing metrics across many simulation cases. It is intentionally decoupled
from any particular project: each case is expected to provide a small "tidy"
metrics file (one row per metric) that any analysis pipeline can produce.

Tidy metric file schema (e.g. ``outputs/metrics.csv``):

    ``scope``    -- ``"plant"`` or a component name (e.g. ``"battery"``).
    ``metric``   -- metric name (e.g. ``"energy_mwh"``, ``"capacity_factor"``,
                    ``"revenue_rt"``, ``"capacity_revenue"``,
                    ``"battery_mileage_soc"``).
    ``value``    -- the raw value over the whole simulation.
    ``unit``     -- a human-readable unit string (optional but recommended).
    ``scaling``  -- one of ``"extensive"``, ``"intensive"``, or ``"annual"``;
                    controls how the value converts between annual and total.
    ``sim_years``-- the simulation length in years (same on every row).

Scaling semantics:

    ``extensive``  The quantity accumulates over time, so the raw ``value`` is
                   a cumulative total for the whole simulation (e.g. energy
                   generated in MWh, total revenue, battery mileage). To get an
                   annual figure, divide by ``sim_years``; the total is just the
                   raw value.
                   annual = value / sim_years; total = value.

    ``intensive``  The quantity is a rate or ratio that does not accumulate —
                   doubling the simulation length does not change the number
                   (e.g. capacity factor, average price). The same value is
                   reported for both annual and total views; no conversion is
                   applied.
                   annual = value; total = value.

    ``annual``     The quantity is already expressed on a per-year basis (e.g.
                   MISO capacity revenue, which is determined by an annual
                   auction). To get the total over the simulation multiply by
                   ``sim_years``; the annual figure is just the raw value.
                   annual = value; total = value * sim_years.
"""

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_COLUMNS = {"case", "scope", "metric", "value", "scaling", "sim_years"}


class ScenarioComparison:
    """Compare metrics across multiple simulation cases.

    Each metric row carries a ``scaling`` tag that describes how its raw
    simulation value should be interpreted when converting between an annual
    figure and the full-simulation total:

    * ``"extensive"`` — cumulative quantities (energy, revenue, mileage).
      The stored value is the simulation total; divide by ``sim_years`` for
      the annual equivalent.
    * ``"intensive"`` — rates or ratios (capacity factor, average price).
      The value is the same regardless of simulation length; no conversion
      is applied.
    * ``"annual"`` — quantities already expressed per year (e.g. MISO
      capacity revenue). Multiply by ``sim_years`` to get the simulation
      total.

    Args:
        metrics (pd.DataFrame): Long-format metrics with at least the columns
            ``case``, ``scope``, ``metric``, ``value``, ``scaling``, and
            ``sim_years`` (an optional ``unit`` column is used for plot
            labels).
        case_names (list[str], optional): Order of cases to use in outputs.
            Defaults to the order of first appearance in ``metrics``.
        display_names (Sequence[str], optional): Human-readable labels aligned
            to ``case_names``, used as the row index in comparison tables.
            Defaults to ``case_names``.
    """

    def __init__(
        self,
        metrics: pd.DataFrame,
        case_names: Sequence[str] | None = None,
        display_names: Sequence[str] | None = None,
    ):
        """Initialize from a long-format metrics DataFrame."""
        missing = REQUIRED_COLUMNS - set(metrics.columns)
        if missing:
            raise ValueError(f"metrics is missing required columns: {sorted(missing)}")
        self.metrics = metrics.copy()
        if "unit" not in self.metrics.columns:
            self.metrics["unit"] = ""
        if case_names is not None:
            self.case_names = list(case_names)
        else:
            self.case_names = list(dict.fromkeys(self.metrics["case"]))
        if display_names is not None:
            self.display_names = list(display_names)
            if len(self.display_names) != len(self.case_names):
                raise ValueError(
                    "display_names length must match case_names length "
                    f"({len(self.display_names)} != {len(self.case_names)})"
                )
        else:
            self.display_names = list(self.case_names)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_cases(
        cls,
        cases: Sequence[str | Path],
        *,
        metric_file: str = "outputs/metrics.csv",
        case_names: Sequence[str] | None = None,
        display_names: Sequence[str] | None = None,
        run_script: str | Path | None = None,
        run_kwargs: dict | None = None,
    ) -> "ScenarioComparison":
        """Build a comparison by collecting per-case metric files.

        Args:
            cases (Sequence[str | Path]): Case directories to compare.
            metric_file (str, optional): Path of the tidy metrics file relative
                to each case directory. Defaults to ``"outputs/metrics.csv"``.
            case_names (Sequence[str], optional): Names for each case. Defaults
                to the case directory names.
            display_names (Sequence[str], optional): Human-readable labels
                aligned to ``case_names``, used as the row index in comparison
                tables. Defaults to ``case_names``.
            run_script (str | Path, optional): If given, this analysis script
                is copied into each case directory and run (headless) to
                regenerate the metric files before collecting. Defaults to None.
            run_kwargs (dict, optional): Extra keyword arguments forwarded to
                :meth:`run_analysis_in_cases`. Defaults to None.

        Returns:
            ScenarioComparison: A comparison built from the collected files.
        """
        if run_script is not None:
            cls.run_analysis_in_cases(cases, run_script, **(run_kwargs or {}))
        long_df = cls.collect(cases, metric_file=metric_file, case_names=case_names)
        return cls(long_df, case_names=case_names, display_names=display_names)

    @staticmethod
    def run_analysis_in_cases(
        cases: Sequence[str | Path],
        analysis_script: str | Path,
        *,
        python_executable: str | None = None,
        extra_args: Sequence[str] | None = None,
        check: bool = True,
    ) -> list[tuple[str, int]]:
        """Copy an analysis script into each case directory and run it headless.

        The script is copied (if not already present at the destination) into
        each case directory and executed with that directory as the working
        directory, so relative paths inside the script resolve per case.

        Args:
            cases (Sequence[str | Path]): Case directories.
            analysis_script (str | Path): Path to the analysis script to copy
                and run in each case.
            python_executable (str, optional): Python interpreter to use.
                Defaults to the current interpreter (``sys.executable``).
            extra_args (Sequence[str], optional): Extra command-line arguments
                passed to the script. Defaults to None.
            check (bool, optional): If True, raise on the first non-zero exit.
                Defaults to True.

        Returns:
            list[tuple[str, int]]: ``(case_dir, return_code)`` for each case.

        Raises:
            RuntimeError: If ``check`` is True and a case run fails.
        """
        python_executable = python_executable or sys.executable
        analysis_script = Path(analysis_script)
        results: list[tuple[str, int]] = []

        for case in cases:
            case_dir = Path(case)
            dest = case_dir / analysis_script.name
            if analysis_script.resolve() != dest.resolve():
                shutil.copy2(analysis_script, dest)

            cmd = [python_executable, analysis_script.name, *(extra_args or [])]
            proc = subprocess.run(
                cmd, cwd=case_dir, capture_output=True, text=True, check=False
            )
            results.append((str(case_dir), proc.returncode))

            if check and proc.returncode != 0:
                raise RuntimeError(
                    f"Analysis script failed in {case_dir} "
                    f"(exit {proc.returncode}).\n"
                    f"--- stdout ---\n{proc.stdout}\n"
                    f"--- stderr ---\n{proc.stderr}"
                )

        return results

    @staticmethod
    def collect(
        cases: Sequence[str | Path],
        *,
        metric_file: str = "outputs/metrics.csv",
        case_names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read and concatenate tidy metric files across cases.

        Args:
            cases (Sequence[str | Path]): Case directories.
            metric_file (str, optional): Metric file path relative to each case
                directory. Defaults to ``"outputs/metrics.csv"``.
            case_names (Sequence[str], optional): Names for each case. Defaults
                to the case directory names.

        Returns:
            pd.DataFrame: Long-format metrics with an added ``case`` column.

        Raises:
            FileNotFoundError: If no metric files are found.
        """
        names = (
            list(case_names)
            if case_names is not None
            else [Path(c).name for c in cases]
        )
        if len(names) != len(cases):
            raise ValueError("case_names length must match cases length")

        frames: list[pd.DataFrame] = []
        for case, name in zip(cases, names, strict=True):
            path = Path(case) / metric_file
            if not path.exists():
                print(f"Warning: metric file not found, skipping: {path}")
                continue
            df = pd.read_csv(path)
            df["case"] = name
            frames.append(df)

        if not frames:
            raise FileNotFoundError(
                f"No metric files named '{metric_file}' found in the given cases."
            )
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        metrics: str | Sequence[str],
        *,
        period: str = "annual",
        scope: str | Sequence[str] = "plant",
    ) -> pd.DataFrame:
        """Build a wide comparison table (cases x metrics).

        Args:
            metrics (str | Sequence[str]): Metric name(s) to select.
            period (str, optional): ``"annual"`` or ``"total"``. Defaults to
                ``"annual"``.
            scope (str | Sequence[str], optional): A single scope (e.g.
                ``"plant"`` or a component name) for flat columns, ``"all"`` to
                include every scope, or a list of scopes. When more than one
                scope is included the result has a ``(scope, metric)``
                MultiIndex on its columns. Defaults to ``"plant"``.

        Returns:
            pd.DataFrame: Cases (rows) by metrics (columns) of scaled values.
        """
        if isinstance(metrics, str):
            metrics = [metrics]
        if period not in ("annual", "total"):
            raise ValueError(f"period must be 'annual' or 'total', got {period!r}")

        single_scope = isinstance(scope, str) and scope != "all"

        sub = self.metrics[self.metrics["metric"].isin(metrics)].copy()
        if single_scope:
            sub = sub[sub["scope"] == scope]
        elif isinstance(scope, (list, tuple)):
            sub = sub[sub["scope"].isin(scope)]

        # An empty subset (e.g. a scope/metric absent from every case) would
        # make pivot_table return an ill-shaped frame, so build the all-NaN
        # result directly with the correct index and columns.
        if sub.empty:
            if single_scope:
                wide = pd.DataFrame(
                    float("nan"), index=self.case_names, columns=list(metrics)
                )
            else:
                wide = pd.DataFrame(index=pd.Index(self.case_names, name="case"))
        else:
            sub["scaled"] = self._scale(sub, period)
            if single_scope:
                wide = sub.pivot_table(
                    index="case", columns="metric", values="scaled", aggfunc="first"
                )
                wide = wide.reindex(index=self.case_names, columns=list(metrics))
            else:
                wide = sub.pivot_table(
                    index="case",
                    columns=["scope", "metric"],
                    values="scaled",
                    aggfunc="first",
                )
                wide = wide.reindex(index=self.case_names)

        # Relabel rows from case names to display labels (a no-op when no
        # display_names were supplied, since they default to case_names).
        wide = wide.rename(
            index=dict(zip(self.case_names, self.display_names, strict=True))
        )
        wide.index.name = "case"
        return wide

    @staticmethod
    def _scale(sub: pd.DataFrame, period: str) -> pd.Series:
        """Convert raw values to the requested period using the scaling tag.

        Only two of the three scaling types ever require a conversion:

        * ``"extensive"`` needs dividing when the caller wants an annual view,
          because the stored value is the cumulative simulation total.
        * ``"annual"`` needs multiplying when the caller wants a total view,
          because the stored value is already expressed per year.
        * ``"intensive"`` is never modified — rates and ratios are the same
          regardless of simulation length.

        Args:
            sub (pd.DataFrame): Subset of the long metrics frame.
            period (str): ``"annual"`` or ``"total"``.

        Returns:
            pd.Series: Scaled values aligned to ``sub``.
        """
        value = sub["value"].astype(float)
        scaling = sub["scaling"]
        sim_years = sub["sim_years"].astype(float)
        out = value.copy()

        if period == "annual":
            # Extensive values are cumulative totals; divide to get per-year.
            # Intensive (rates/ratios) and annual-tagged values need no change.
            ext = scaling == "extensive"
            out[ext] = value[ext] / sim_years[ext]
        else:  # total
            # Annual-tagged values are stored per year; multiply to get total.
            # Extensive values are already totals; intensive need no change.
            ann = scaling == "annual"
            out[ann] = value[ann] * sim_years[ann]

        return out

    def _unit(self, metric: str, scope: str) -> str:
        """Return the unit string recorded for a metric/scope, if any.

        Args:
            metric (str): Metric name.
            scope (str): Scope name.

        Returns:
            str: The unit string, or an empty string if not found.
        """
        match = self.metrics[
            (self.metrics["metric"] == metric) & (self.metrics["scope"] == scope)
        ]
        if match.empty:
            return ""
        return str(match["unit"].iloc[0])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @staticmethod
    def to_csv(comparison_df: pd.DataFrame, path: str | Path) -> Path:
        """Write a comparison table to CSV.

        Args:
            comparison_df (pd.DataFrame): A table from :meth:`compare`.
            path (str | Path): Output CSV path.

        Returns:
            Path: The path written.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(out)
        print(f"Comparison table saved to: {out}")
        return out

    @staticmethod
    def to_great_table(
        comparison_df: pd.DataFrame,
        *,
        title: str | None = None,
        subtitle: str | None = None,
        decimals: int = 2,
    ):
        """Render a comparison table as a ``great_tables`` table.

        Args:
            comparison_df (pd.DataFrame): A table from :meth:`compare`.
            title (str, optional): Table title. Defaults to None.
            subtitle (str, optional): Table subtitle. Defaults to None.
            decimals (int, optional): Decimal places for numeric formatting.
                Defaults to 2.

        Returns:
            great_tables.GT: The formatted table object.

        Raises:
            ImportError: If ``great_tables`` is not installed.
        """
        try:
            from great_tables import GT
        except ImportError as exc:
            raise ImportError(
                "great_tables is required for to_great_table(). Install it with "
                "'pip install great-tables'."
            ) from exc

        df = comparison_df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [":".join(str(level) for level in col) for col in df.columns]
        df = df.reset_index()

        gt = GT(df, rowname_col="case")
        if title is not None:
            gt = gt.tab_header(title=title, subtitle=subtitle or "")
        num_cols = [c for c in df.columns if c != "case"]
        if num_cols:
            gt = gt.fmt_number(columns=num_cols, decimals=decimals)
        return gt

    def plot_metric(
        self,
        metric: str,
        *,
        period: str = "annual",
        scope: str = "plant",
        kind: str = "bar",
        ax=None,
        **kwargs,
    ):
        """Plot a single metric across cases.

        Args:
            metric (str): Metric name to plot.
            period (str, optional): ``"annual"`` or ``"total"``. Defaults to
                ``"annual"``.
            scope (str, optional): Scope to plot (``"plant"`` or a component
                name). Defaults to ``"plant"``.
            kind (str, optional): ``"bar"`` or ``"line"``. Defaults to
                ``"bar"``.
            ax (matplotlib.axes.Axes, optional): Existing axis to draw on.
                Defaults to None (a new figure/axis is created).
            **kwargs: Forwarded to the matplotlib plotting call.

        Returns:
            tuple: ``(fig, ax)``.
        """
        wide = self.compare(metric, period=period, scope=scope)
        values = wide[metric] if metric in wide.columns else wide.iloc[:, 0]

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        x = range(len(wide.index))
        if kind == "bar":
            ax.bar(x, values.to_numpy(), **kwargs)
        elif kind == "line":
            kwargs.setdefault("marker", "o")
            ax.plot(x, values.to_numpy(), **kwargs)
        else:
            raise ValueError(f"kind must be 'bar' or 'line', got {kind!r}")

        unit = self._unit(metric, scope)
        ylabel = f"{metric} ({unit})" if unit else metric
        period_label = "annual" if period == "annual" else "total"
        ax.set_ylabel(f"{ylabel} [{period_label}]")
        ax.set_xticks(list(x))
        ax.set_xticklabels(wide.index, rotation=45, ha="right")
        ax.grid(True, axis="y")
        fig.tight_layout()
        return fig, ax

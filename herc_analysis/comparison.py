"""L5 -- the single cross-scenario comparison engine.

``Comparison`` replaces both ``TotalMetrics``' list-mode and the standalone
``ScenarioComparison``. It always takes exactly one input -- the long-format
metrics table (the :class:`~herc_analysis.metrics.MetricSet` shape) -- and the
two named constructors ``from_scenarios`` / ``from_cases`` just produce that
same table from a different source (live ``Scenario`` objects, or saved
``metrics.csv`` files) before handing it to ``__init__``.

The scaling semantics (``extensive`` / ``intensive`` / ``annual``) carry over
unchanged from the original ``ScenarioComparison``; the only schema change is
``scope`` -> ``entity`` plus the temporal ``resolution`` / ``period`` axis.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"case", "entity", "metric", "value", "scaling", "sim_years"}


class Comparison:
    """Compare metrics across runs. The single comparison engine.

    Args:
        metrics (pd.DataFrame): Long-format metrics with at least the columns
            ``case``, ``entity``, ``metric``, ``value``, ``scaling`` and
            ``sim_years`` (optional ``unit`` / ``resolution`` / ``period``).
        case_names (Sequence[str], optional): Case order for outputs. Defaults
            to first-appearance order in ``metrics``.
        display_names (Sequence[str], optional): Row labels aligned to
            ``case_names``. Defaults to ``case_names``.
    """

    def __init__(
        self,
        metrics: pd.DataFrame,
        *,
        case_names: Sequence[str] | None = None,
        display_names: Sequence[str] | None = None,
    ):
        missing = REQUIRED_COLUMNS - set(metrics.columns)
        if missing:
            raise ValueError(f"metrics is missing required columns: {sorted(missing)}")

        self.metrics = metrics.copy()
        # Make optional axes/labels robust so minimal long files still work.
        if "unit" not in self.metrics.columns:
            self.metrics["unit"] = ""
        if "resolution" not in self.metrics.columns:
            self.metrics["resolution"] = "total"
        if "period" not in self.metrics.columns:
            self.metrics["period"] = "total"

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
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_scenarios(
        cls,
        scenarios: Sequence,
        names: Sequence[str] | None = None,
        *,
        display_names: Sequence[str] | None = None,
    ) -> Comparison:
        """Build from live ``Scenario`` objects by stacking their metric sets.

        Args:
            scenarios (Sequence[Scenario]): The scenarios to compare.
            names (Sequence[str], optional): Case names. Defaults to each
                scenario's ``name``.
            display_names (Sequence[str], optional): Row labels. Defaults to
                ``names``.

        Returns:
            Comparison: A comparison over the stacked long-format metrics.
        """
        names = list(names) if names is not None else [s.name for s in scenarios]
        frames = []
        for s, n in zip(scenarios, names, strict=True):
            long = s.metric_set.to_long()
            long["case"] = n
            frames.append(long)
        return cls(
            pd.concat(frames, ignore_index=True),
            case_names=names,
            display_names=display_names,
        )

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
    ) -> Comparison:
        """Build by collecting per-case ``metrics.csv`` files.

        Args:
            cases (Sequence[str | Path]): Case directories to compare.
            metric_file (str): Metric file path relative to each case directory.
                Defaults to ``"outputs/metrics.csv"``.
            case_names (Sequence[str], optional): Names for each case. Defaults
                to the directory names.
            display_names (Sequence[str], optional): Row labels. Defaults to
                ``case_names``.
            run_script (str | Path, optional): If given, copy and run this
                analysis script in each case directory first (to regenerate the
                metric files). Defaults to None.
            run_kwargs (dict, optional): Extra kwargs for
                :meth:`run_analysis_in_cases`. Defaults to None.

        Returns:
            Comparison: A comparison built from the collected files.
        """
        if run_script is not None:
            cls.run_analysis_in_cases(cases, run_script, **(run_kwargs or {}))
        long_df = cls.collect(cases, metric_file=metric_file, case_names=case_names)
        return cls(long_df, case_names=case_names, display_names=display_names)

    @staticmethod
    def collect(
        cases: Sequence[str | Path],
        *,
        metric_file: str = "outputs/metrics.csv",
        case_names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read and concatenate per-case metric files into one long frame.

        Args:
            cases (Sequence[str | Path]): Case directories.
            metric_file (str): Metric file path relative to each case directory.
            case_names (Sequence[str], optional): Names for each case. Defaults
                to the directory names.

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

        Args:
            cases (Sequence[str | Path]): Case directories.
            analysis_script (str | Path): Script to copy and run per case.
            python_executable (str, optional): Interpreter. Defaults to the
                current one.
            extra_args (Sequence[str], optional): Extra CLI args. Defaults None.
            check (bool): Raise on the first non-zero exit. Defaults True.

        Returns:
            list[tuple[str, int]]: ``(case_dir, return_code)`` per case.

        Raises:
            RuntimeError: If ``check`` and a case run fails.
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

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def table(
        self,
        metrics: str | Sequence[str],
        *,
        view: str = "per_year",
        entity: str | Sequence[str] = "plant",
        resolution: str = "total",
    ) -> pd.DataFrame:
        """Build a wide comparison table (cases x metrics) of scaled values.

        Args:
            metrics (str | Sequence[str]): Metric name(s) to select.
            view (str): The scaling view -- ``"per_year"`` (extensive totals
                divided by ``sim_years``) or ``"cumulative"`` (the whole-run
                totals; annual-tagged values multiplied up). Defaults to
                ``"per_year"``.
            entity (str | Sequence[str]): A single entity (``"plant"`` or a
                component) for flat columns, ``"all"`` for every entity, or a
                list. Multiple entities produce a ``(entity, metric)`` column
                MultiIndex. Defaults to ``"plant"``.
            resolution (str): Temporal resolution to read rows from
                (``"total"`` / ``"annual"`` / ``"yearly"`` / ``"monthly"``).
                Defaults to ``"total"``.

        Returns:
            pd.DataFrame: Cases (rows) by metrics (columns) of scaled values.
        """
        if isinstance(metrics, str):
            metrics = [metrics]
        if view not in ("per_year", "cumulative"):
            raise ValueError(f"view must be 'per_year' or 'cumulative', got {view!r}")

        single = isinstance(entity, str) and entity != "all"

        sub = self.metrics[
            (self.metrics["metric"].isin(metrics))
            & (self.metrics["resolution"] == resolution)
        ].copy()
        if single:
            sub = sub[sub["entity"] == entity]
        elif isinstance(entity, (list, tuple)):
            sub = sub[sub["entity"].isin(entity)]

        if sub.empty:
            if single:
                wide = pd.DataFrame(
                    float("nan"), index=self.case_names, columns=list(metrics)
                )
            else:
                wide = pd.DataFrame(index=pd.Index(self.case_names, name="case"))
        else:
            sub["scaled"] = self._scale(sub, view)
            if single:
                wide = sub.pivot_table(
                    index="case", columns="metric", values="scaled", aggfunc="first"
                )
                wide = wide.reindex(index=self.case_names, columns=list(metrics))
            else:
                wide = sub.pivot_table(
                    index="case",
                    columns=["entity", "metric"],
                    values="scaled",
                    aggfunc="first",
                )
                wide = wide.reindex(index=self.case_names)

        wide = wide.rename(
            index=dict(zip(self.case_names, self.display_names, strict=True))
        )
        wide.index.name = "case"
        return wide

    @staticmethod
    def _scale(sub: pd.DataFrame, view: str) -> pd.Series:
        """Convert whole-run total values to the requested view via the scaling tag.

        The per_year/cumulative conversion only applies to ``resolution="total"``
        rows (cumulative totals). Already-bucketed resolutions
        (``annual`` / ``yearly`` / ``monthly``) are at their stated granularity
        and are returned unchanged. For total rows: ``extensive`` divides by
        ``sim_years`` for ``per_year``; ``annual`` multiplies by ``sim_years``
        for ``cumulative``; ``intensive`` is never modified.
        """
        value = sub["value"].astype(float)
        if (sub["resolution"] != "total").any():
            return value

        scaling = sub["scaling"]
        sim_years = sub["sim_years"].astype(float)
        out = value.copy()
        if view == "per_year":
            ext = scaling == "extensive"
            out[ext] = value[ext] / sim_years[ext]
        else:  # cumulative
            ann = scaling == "annual"
            out[ann] = value[ann] * sim_years[ann]
        return out

    def _unit(self, metric: str, entity: str) -> str:
        """Return the unit string recorded for a metric/entity, if any."""
        match = self.metrics[
            (self.metrics["metric"] == metric) & (self.metrics["entity"] == entity)
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
            comparison_df (pd.DataFrame): A table from :meth:`table`.
            path (str | Path): Output CSV path.

        Returns:
            Path: The path written.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(out)
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
            comparison_df (pd.DataFrame): A table from :meth:`table`.
            title (str, optional): Table title. Defaults to None.
            subtitle (str, optional): Table subtitle. Defaults to None.
            decimals (int): Decimal places. Defaults to 2.

        Returns:
            great_tables.GT: The formatted table.

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

    def plot(
        self,
        metric: str,
        *,
        view: str = "per_year",
        entity: str = "plant",
        resolution: str = "total",
        kind: str = "bar",
        ax=None,
        **kwargs,
    ):
        """Plot a single metric across cases.

        Args:
            metric (str): Metric name to plot.
            view (str): ``"per_year"`` or ``"cumulative"``. Defaults to
                ``"per_year"``.
            entity (str): Entity to plot. Defaults to ``"plant"``.
            resolution (str): Temporal resolution. Defaults to ``"total"``.
            kind (str): ``"bar"`` or ``"line"``. Defaults to ``"bar"``.
            ax (matplotlib.axes.Axes, optional): Existing axis. Defaults to None.
            **kwargs: Forwarded to the matplotlib call.

        Returns:
            tuple: ``(fig, ax)``.
        """
        import matplotlib.pyplot as plt

        wide = self.table(metric, view=view, entity=entity, resolution=resolution)
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

        unit = self._unit(metric, entity)
        ylabel = f"{metric} ({unit})" if unit else metric
        ax.set_ylabel(f"{ylabel} [{view}]")
        ax.set_xticks(list(x))
        ax.set_xticklabels(wide.index, rotation=45, ha="right")
        ax.grid(True, axis="y")
        fig.tight_layout()
        return fig, ax

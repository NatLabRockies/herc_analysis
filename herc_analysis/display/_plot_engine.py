"""PlotHerculesOutput class for generic, signal-driven Plotly plotting."""

import os
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from herc_analysis.constants import PTC_PRICE, SignalSubplot
from herc_analysis.display.colors import (
    INFRASTRUCTURE_COLORS,
    get_component_color,
)
from herc_analysis.timeseries import interpolate_df


class PlotHerculesOutput:
    """Interactive Plotly plotting for one or more Hercules scenarios.

    Standard subplots (plant power, market) are toggled via booleans. When
    both are enabled, custom signal panels are stacked above them, with plant
    power above market (market is the bottom row with the shared x-axis
    label).
    Arbitrary signal subplots are specified with :class:`SignalSubplot`.
    """

    # Columns that belong to a component but are not useful to plot directly
    _SKIP_SIGNAL_SUFFIXES = (
        "_energy_mwh",
        "_revenue_rt",
        "_revenue_da",
        "_revenue_rt_discharge",
        "_revenue_rt_charge",
    )

    def __init__(
        self,
        output_analysis,
        scenario_names: list[str] | None = None,
    ):
        """Initialize with one or more OutputAnalysis objects.

        Args:
            output_analysis (OutputAnalysis | list[OutputAnalysis]): Single or
                list of OutputAnalysis objects.
            scenario_names (list[str], optional): Scenario labels.
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

        self.print_available_signals()

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def print_available_signals(self):
        """Print signals available for use in :class:`SignalSubplot`."""
        ref_oa = self.output_analyses[0]
        df = ref_oa.df

        print("\nAvailable signals for plotting:")
        for comp in ref_oa.components:
            cols = self._component_signal_columns(comp.name, df)
            print(f"  {comp.name} ({comp.category}, {comp.component_type}):")
            for c in cols:
                print(f"    - {c}")

        plant_cols = [
            c for c in ("total_plant_power_mw", "lmp_rt", "lmp_da") if c in df.columns
        ]
        if plant_cols:
            print(f"  Plant-level: {', '.join(plant_cols)}")
        print()

    def _component_signal_columns(self, name: str, df: pd.DataFrame) -> list[str]:
        """Return plottable column names for a component.

        Args:
            name (str): Component name.
            df (pd.DataFrame): Reference dataframe.

        Returns:
            list[str]: Sorted column names.
        """
        cols: list[str] = []
        for c in sorted(df.columns):
            if not (c.startswith(f"{name}_") or c.startswith(f"{name}.")):
                continue
            if c.endswith(".power"):
                continue
            if any(c.endswith(s) for s in self._SKIP_SIGNAL_SUFFIXES):
                continue
            cols.append(c)
        return cols

    def component_power_subplot(self, component_name: str) -> SignalSubplot:
        """Create a SignalSubplot for a component's power (and setpoint if available).

        Args:
            component_name (str): Name of the component.

        Returns:
            SignalSubplot: Configured subplot spec.
        """
        ref_df = self.output_analyses[0].df
        cols = [f"{component_name}_power_mw"]
        sp_col = f"{component_name}_power_setpoint_mw"
        if sp_col in ref_df.columns:
            cols.append(sp_col)
        return SignalSubplot(
            columns=cols,
            title=f"{component_name} Power",
            y_label="Power (MW)",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_interactive(
        self,
        plot_dt: int = 60,
        height_per_subplot: int = 300,
        save_file: str | None = None,
        open_browser: bool = True,
        show_plant_power: bool = True,
        show_market: bool = True,
        show_negative_ptc_line: bool = True,
        shade_price_area: bool = True,
        show_interconnect_limit: bool = True,
        date_range: list[str] | None = None,
        signal_subplots: list[SignalSubplot] | None = None,
        da_only: bool = False,
    ):
        """Create interactive Plotly figure.

        Args:
            plot_dt (int, optional): Downsample interval in seconds.
                Defaults to 60.
            height_per_subplot (int, optional): Pixel height per subplot.
                Defaults to 300.
            save_file (str, optional): HTML output path. Defaults to None.
            open_browser (bool, optional): Open file in browser.
                Defaults to True.
            show_plant_power (bool, optional): Include stacked plant power
                subplot. When ``show_market`` is also True, plant power sits
                above market (second row from the bottom). If market is off,
                plant power is the bottom subplot. Defaults to True.
            show_market (bool, optional): Include market price subplot. When
                enabled, it is always the bottom subplot. Defaults to True.
            show_negative_ptc_line (bool, optional): Show -PTC reference.
                Defaults to True.
            shade_price_area (bool, optional): Shade positive/negative price.
                Defaults to True.
            show_interconnect_limit (bool, optional): Show interconnect line.
                Defaults to True.
            date_range (list[str], optional): ['YYYY-MM-DD','YYYY-MM-DD'].
                Defaults to None.
            signal_subplots (list[SignalSubplot], optional): Custom signal
                subplots. Defaults to None.
            da_only (bool, optional): Show only the day-ahead price in
                the market subplot, with a fill-to-zero shading on the DA
                line. The RT trace is hidden. Defaults to False.

        Returns:
            plotly.graph_objects.Figure: The figure.
        """
        ref_oa = self.output_analyses[0]

        downsampled_dfs = []
        for oa in self.output_analyses:
            df = oa.df.copy()
            if date_range is not None:
                df = self._filter_by_date_range(df, date_range)
            df = self._downsample_data(df, plot_dt)
            downsampled_dfs.append(df)

        subplot_plan = self._build_subplot_plan(
            show_plant_power, show_market, signal_subplots
        )

        n_subplots = len(subplot_plan)
        if n_subplots == 0:
            print("Warning: no subplots to show.")
            return go.Figure()

        titles = [s["title"] for s in subplot_plan]

        fig = make_subplots(
            rows=n_subplots,
            cols=1,
            shared_xaxes=True,
            subplot_titles=titles,
            vertical_spacing=0.03,
        )

        subplot_to_legend = {}
        for i in range(1, n_subplots + 1):
            subplot_to_legend[i] = "legend" if i == 1 else f"legend{i}"

        time_col, x_label = ref_oa._get_time_axis_info()

        cat_counters: dict[str, int] = {}
        comp_colors: dict[str, str] = {}
        for comp in ref_oa.components:
            idx = cat_counters.get(comp.category, 0)
            comp_colors[comp.name] = get_component_color(comp.category, idx)
            cat_counters[comp.category] = idx + 1

        for row_idx, spec in enumerate(subplot_plan, start=1):
            renderer = spec["renderer"]
            renderer(
                fig=fig,
                dfs=downsampled_dfs,
                time_col=time_col,
                subplot_row=row_idx,
                ref_oa=ref_oa,
                subplot_to_legend=subplot_to_legend,
                comp_colors=comp_colors,
                spec=spec,
                show_interconnect_limit=show_interconnect_limit,
                show_negative_ptc_line=show_negative_ptc_line,
                shade_price_area=shade_price_area,
                da_only=da_only,
            )

        # Scenario legend for multi-scenario
        if self.is_multi_scenario:
            sleg = f"legend{n_subplots + 1}" if n_subplots > 0 else "legend"
            for i, sname in enumerate(self.scenario_names):
                ms = self._get_marker_symbol(i)
                mode = "lines+markers" if ms else "lines"
                mkw = {"marker": {"size": 4, "symbol": ms}} if ms else {}
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode=mode,
                        name=sname,
                        line={"color": "gray", "width": 2},
                        opacity=self._get_alpha(i),
                        showlegend=True,
                        legend=sleg,
                        legendgroup="scenarios",
                        **mkw,
                    )
                )

        # Layout
        legend_pos = self._legend_positions(n_subplots, 0.03)
        layout_kw: dict = {
            "height": n_subplots * height_per_subplot,
            "title_text": "Interactive Simulation Output Analysis"
            + (" (Multi-Scenario)" if self.is_multi_scenario else ""),
            "title_x": 0.5,
            "showlegend": True,
            "hovermode": "x unified",
        }
        for i in range(1, n_subplots + 1):
            lname = subplot_to_legend[i]
            pos = legend_pos[i]
            layout_kw[lname] = {
                "orientation": "v",
                "yanchor": "top",
                "y": pos["y"],
                "xanchor": "right",
                "x": pos["x"],
                "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": "rgba(0,0,0,0.2)",
                "borderwidth": 1,
                "groupclick": "toggleitem",
                "font": {"size": 9},
                "itemsizing": "constant",
                "tracegroupgap": 5,
            }
        if self.is_multi_scenario:
            sleg = f"legend{n_subplots + 1}" if n_subplots > 0 else "legend"
            layout_kw[sleg] = {
                "orientation": "h",
                "yanchor": "bottom",
                "y": -0.05,
                "xanchor": "center",
                "x": 0.5,
                "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": "rgba(0,0,0,0.2)",
                "borderwidth": 1,
                "font": {"size": 10},
                "tracegroupgap": 10,
            }

        fig.update_layout(**layout_kw)
        fig.update_xaxes(title_text=x_label, row=n_subplots, col=1)
        for i in range(1, n_subplots + 1):
            fig.update_xaxes(showgrid=True, row=i, col=1)
            fig.update_yaxes(showgrid=True, row=i, col=1)

        if save_file is not None:
            fig.write_html(save_file)
            print(f"Interactive plot saved as: {save_file}")
            if os.path.exists(save_file):
                mb = os.path.getsize(save_file) / (1024 * 1024)
                print(f"File size: {mb:.1f}MB")
            if open_browser:
                try:
                    webbrowser.open(f"file://{os.path.abspath(save_file)}")
                except Exception as e:
                    print(f"Note: Could not auto-open browser ({e})")

        return fig

    # ------------------------------------------------------------------
    # Subplot plan builder
    # ------------------------------------------------------------------

    def _build_subplot_plan(
        self,
        show_plant_power: bool,
        show_market: bool,
        signal_subplots: list[SignalSubplot] | None,
    ) -> list[dict]:
        """Build ordered list of subplot specs.

        Row order is top to bottom: custom ``signal_subplots`` first, then
        plant power (if enabled), then market (if enabled). Market is always
        the bottom row when it is included; plant power is the lowest row when
        market is off.

        Args:
            show_plant_power (bool): Include plant power subplot.
            show_market (bool): Include market subplot.
            signal_subplots (list[SignalSubplot], optional): Custom signal
                subplots.

        Returns:
            list[dict]: Each dict has 'id', 'title', 'renderer', plus
                extra keys consumed by the renderer.
        """
        plan: list[dict] = []

        for idx, ss in enumerate(signal_subplots or []):
            cols = [ss.columns] if isinstance(ss.columns, str) else list(ss.columns)
            title = ss.title or ", ".join(cols)
            plan.append(
                {
                    "id": f"signal_{idx}",
                    "title": title,
                    "renderer": self._render_signal,
                    "signal_spec": ss,
                    "columns": cols,
                }
            )

        if show_plant_power:
            plan.append(
                {
                    "id": "plant_power",
                    "title": "Plant Power",
                    "renderer": self._render_plant_power,
                }
            )

        if show_market:
            plan.append(
                {
                    "id": "market",
                    "title": "Market Price",
                    "renderer": self._render_market,
                }
            )

        return plan

    # ------------------------------------------------------------------
    # Renderers
    # ------------------------------------------------------------------

    def _render_signal(
        self,
        *,
        fig,
        dfs,
        time_col,
        subplot_row,
        subplot_to_legend,
        comp_colors,
        spec,
        **_kw,
    ):
        """Generic renderer for one or more DataFrame columns.

        Handles scaling, auto-coloring, and optional reference lines.
        """
        ss: SignalSubplot = spec["signal_spec"]
        cols: list[str] = spec["columns"]
        scale = ss.scale

        default_palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ]

        for i, (df, sname) in enumerate(zip(dfs, self.scenario_names, strict=False)):
            w, a, ms = self._style(i)
            for col_idx, col in enumerate(cols):
                if col not in df.columns:
                    continue

                color = self._resolve_color(col, col_idx, comp_colors, default_palette)

                col_label = col.rsplit(".", 1)[-1] if "." in col else col
                if self.is_multi_scenario:
                    label = f"{sname} - {col_label}"
                else:
                    label = col_label

                y = df[col] * scale
                mode = "lines+markers" if ms else "lines"
                mkw = {}
                if ms:
                    mkw["marker"] = {
                        "size": 4,
                        "symbol": ms,
                        "opacity": a,
                    }
                fig.add_trace(
                    go.Scatter(
                        x=df[time_col],
                        y=y,
                        mode=mode,
                        name=label,
                        line={"color": color, "width": w},
                        opacity=a,
                        hovertemplate=f"{col_label}: %{{y:.2f}}<extra></extra>",
                        showlegend=True,
                        legend=subplot_to_legend.get(subplot_row, "legend"),
                        **mkw,
                    ),
                    row=subplot_row,
                    col=1,
                )

        labels = ss.reference_line_labels or []
        for idx, ref_val in enumerate(ss.reference_lines or []):
            label = labels[idx] if idx < len(labels) else None
            fig.add_hline(
                y=ref_val,
                line_dash="dash",
                line_color=INFRASTRUCTURE_COLORS["reference_line"],
                opacity=0.5,
                annotation_text=label,
                row=subplot_row,
                col=1,
            )

        if ss.y_label:
            fig.update_yaxes(title_text=ss.y_label, row=subplot_row, col=1)

    @staticmethod
    def _resolve_color(
        col: str,
        col_idx: int,
        comp_colors: dict[str, str],
        default_palette: list[str],
    ) -> str:
        """Pick a colour for a signal column.

        Uses the component colour when the column belongs to a known
        component and is the only column, otherwise cycles a palette.
        """
        for comp_name, color in comp_colors.items():
            if col.startswith(f"{comp_name}_") or col.startswith(f"{comp_name}."):
                if col_idx == 0:
                    return color
                break
        return default_palette[col_idx % len(default_palette)]

    def _render_plant_power(
        self,
        *,
        fig,
        dfs,
        time_col,
        subplot_row,
        ref_oa,
        subplot_to_legend,
        comp_colors,
        show_interconnect_limit,
        **_kw,
    ):
        if self.is_multi_scenario:
            for i, (df, sname, oa) in enumerate(
                zip(dfs, self.scenario_names, self.output_analyses, strict=False)
            ):
                w, a, ms = self._style(i)
                label = f"{sname} - Total Power"
                self._trace(
                    fig,
                    df,
                    time_col,
                    "total_plant_power_mw",
                    label,
                    INFRASTRUCTURE_COLORS["total_power"],
                    subplot_row,
                    subplot_to_legend,
                    w,
                    a,
                    ms,
                    "Total Power: %{y:.1f} MW<extra></extra>",
                )
                if i == 0 and show_interconnect_limit:
                    fig.add_hline(
                        y=oa.interconnect_mw,
                        line_dash="dash",
                        line_color=INFRASTRUCTURE_COLORS["interconnect"],
                        opacity=0.5,
                        row=subplot_row,
                        col=1,
                    )
        else:
            df = dfs[0]
            oa = self.output_analyses[0]
            legend_name = subplot_to_legend.get(subplot_row, "legend")
            cumulative = pd.Series(0.0, index=df.index)

            for comp in oa.components:
                col = f"{comp.name}_power_mw"
                if col not in df.columns:
                    continue
                cumulative = cumulative + df[col]
                color = comp_colors.get(comp.name, "#888888")
                r, g, b = (
                    int(color[1:3], 16),
                    int(color[3:5], 16),
                    int(color[5:7], 16),
                )
                fig.add_trace(
                    go.Scatter(
                        x=df[time_col],
                        y=cumulative,
                        mode="lines",
                        fill="tonexty",
                        name=comp.name,
                        line={"color": color},
                        fillcolor=f"rgba({r},{g},{b},0.4)",
                        customdata=df[col],
                        hovertemplate=(
                            f"{comp.name}: %{{customdata:.1f}} MW<extra></extra>"
                        ),
                        legendgroup="plant_power",
                        legend=legend_name,
                    ),
                    row=subplot_row,
                    col=1,
                )

            fig.add_trace(
                go.Scatter(
                    x=df[time_col],
                    y=df["total_plant_power_mw"],
                    mode="lines",
                    name="Total Plant Power",
                    line={
                        "color": INFRASTRUCTURE_COLORS["total_power"],
                        "width": 2,
                    },
                    hovertemplate="Total: %{y:.1f} MW<extra></extra>",
                    legendgroup="plant_power",
                    legend=legend_name,
                ),
                row=subplot_row,
                col=1,
            )

            if show_interconnect_limit:
                fig.add_hline(
                    y=oa.interconnect_mw,
                    line_dash="dash",
                    line_color=INFRASTRUCTURE_COLORS["interconnect"],
                    annotation_text="Interconnect",
                    row=subplot_row,
                    col=1,
                )

        fig.update_yaxes(title_text="Power (MW)", row=subplot_row, col=1)

    def _render_market(
        self,
        *,
        fig,
        dfs,
        time_col,
        subplot_row,
        subplot_to_legend,
        show_negative_ptc_line,
        shade_price_area,
        da_only=False,
        **_kw,
    ):
        for i, (df, sname) in enumerate(zip(dfs, self.scenario_names, strict=False)):
            w, a, ms = self._style(i)
            legend_name = subplot_to_legend.get(subplot_row, "legend")

            if da_only:
                # Fill-to-zero shading on the DA line
                if i == 0 and "lmp_da" in df.columns:
                    pos_y = df["lmp_da"].clip(lower=0)
                    fig.add_trace(
                        go.Scatter(
                            x=df[time_col],
                            y=pos_y,
                            mode="lines",
                            fill="tozeroy",
                            name="Positive DA Price",
                            line={"color": "rgba(0,128,0,0)", "width": 0},
                            fillcolor="rgba(0,255,0,0.2)",
                            showlegend=False,
                            hoverinfo="skip",
                            legend=legend_name,
                        ),
                        row=subplot_row,
                        col=1,
                    )
                    neg_y = df["lmp_da"].clip(upper=0)
                    fig.add_trace(
                        go.Scatter(
                            x=df[time_col],
                            y=neg_y,
                            mode="lines",
                            fill="tozeroy",
                            name="Negative DA Price",
                            line={"color": "rgba(128,0,0,0)", "width": 0},
                            fillcolor="rgba(255,0,0,0.2)",
                            showlegend=False,
                            hoverinfo="skip",
                            legend=legend_name,
                        ),
                        row=subplot_row,
                        col=1,
                    )
                da_label = f"{sname} - DA" if self.is_multi_scenario else "DA Price"
                self._trace(
                    fig,
                    df,
                    time_col,
                    "lmp_da",
                    da_label,
                    INFRASTRUCTURE_COLORS["market_da"],
                    subplot_row,
                    subplot_to_legend,
                    w,
                    a,
                    ms,
                    "DA: %{y:.2f} $/MWh<extra></extra>",
                )
                continue

            if shade_price_area and i == 0:
                pos_y = df["lmp_rt"].clip(lower=0)
                fig.add_trace(
                    go.Scatter(
                        x=df[time_col],
                        y=pos_y,
                        mode="lines",
                        fill="tozeroy",
                        name="Positive Price",
                        line={"color": "rgba(0,128,0,0)", "width": 0},
                        fillcolor="rgba(0,255,0,0.2)",
                        showlegend=False,
                        hoverinfo="skip",
                        legend=legend_name,
                    ),
                    row=subplot_row,
                    col=1,
                )
                neg_y = df["lmp_rt"].clip(upper=0)
                fig.add_trace(
                    go.Scatter(
                        x=df[time_col],
                        y=neg_y,
                        mode="lines",
                        fill="tozeroy",
                        name="Negative Price",
                        line={"color": "rgba(128,0,0,0)", "width": 0},
                        fillcolor="rgba(255,0,0,0.2)",
                        showlegend=False,
                        hoverinfo="skip",
                        legend=legend_name,
                    ),
                    row=subplot_row,
                    col=1,
                )

            rt_label = f"{sname} - RT" if self.is_multi_scenario else "RT Price"
            self._trace(
                fig,
                df,
                time_col,
                "lmp_rt",
                rt_label,
                INFRASTRUCTURE_COLORS["market_rt"],
                subplot_row,
                subplot_to_legend,
                w,
                a,
                ms,
                "RT: %{y:.2f} $/MWh<extra></extra>",
            )

            da_label = f"{sname} - DA" if self.is_multi_scenario else "DA Price"
            self._trace(
                fig,
                df,
                time_col,
                "lmp_da",
                da_label,
                INFRASTRUCTURE_COLORS["market_da"],
                subplot_row,
                subplot_to_legend,
                w,
                a * 0.7,
                ms,
                "DA: %{y:.2f} $/MWh<extra></extra>",
            )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color=INFRASTRUCTURE_COLORS["reference_line"],
            row=subplot_row,
            col=1,
        )
        if show_negative_ptc_line:
            fig.add_hline(
                y=-PTC_PRICE,
                line_dash="dash",
                line_color=INFRASTRUCTURE_COLORS["ptc_revenue"],
                annotation_text="Negative PTC",
                row=subplot_row,
                col=1,
            )
        fig.update_yaxes(title_text="Price ($/MWh)", row=subplot_row, col=1)

    # ------------------------------------------------------------------
    # Trace helper
    # ------------------------------------------------------------------

    def _trace(
        self,
        fig,
        df,
        time_col,
        col,
        name,
        color,
        subplot_row,
        subplot_to_legend,
        width,
        alpha,
        marker_symbol,
        hover,
    ):
        if col not in df.columns:
            return
        mode = "lines+markers" if marker_symbol else "lines"
        mkw = {}
        if marker_symbol:
            mkw["marker"] = {
                "size": 4,
                "symbol": marker_symbol,
                "opacity": alpha,
            }
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[col],
                mode=mode,
                name=name,
                line={"color": color, "width": width},
                opacity=alpha,
                hovertemplate=hover,
                showlegend=True,
                legend=subplot_to_legend.get(subplot_row, "legend"),
                **mkw,
            ),
            row=subplot_row,
            col=1,
        )

    # ------------------------------------------------------------------
    # Style helpers
    # ------------------------------------------------------------------

    def _style(self, scenario_idx):
        return (
            2,
            self._get_alpha(scenario_idx),
            (self._get_marker_symbol(scenario_idx) if self.is_multi_scenario else None),
        )

    def _get_alpha(self, idx):
        if not self.is_multi_scenario:
            return 1.0
        if idx == 0:
            return 1.0
        return [0.85, 0.7, 0.55][(idx - 1) % 3]

    @staticmethod
    def _get_marker_symbol(idx):
        if idx == 0:
            return None
        symbols = [
            "circle",
            "square",
            "diamond",
            "triangle-up",
            "x",
            "cross",
        ]
        return symbols[(idx - 1) % len(symbols)]

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_by_date_range(df, date_range):
        if len(date_range) != 2:
            raise ValueError("date_range must be ['YYYY-MM-DD', 'YYYY-MM-DD']")
        if "time_utc" not in df.columns:
            raise ValueError("date_range requires time_utc column")
        start = pd.to_datetime(date_range[0])
        end = pd.to_datetime(date_range[1])
        if not pd.api.types.is_datetime64_any_dtype(df["time_utc"]):
            df = df.copy()
            df["time_utc"] = pd.to_datetime(df["time_utc"])
        mask = (df["time_utc"].dt.date >= start.date()) & (
            df["time_utc"].dt.date <= end.date()
        )
        return df[mask].copy()

    @staticmethod
    def _downsample_data(df, plot_dt):
        if df["time"].dtype == "object":
            df["time"] = df["time"].astype(float)
        new_time = np.arange(df["time"].min(), df["time"].max(), plot_dt)
        return interpolate_df(
            df, new_time, interpolation_method="instantaneous_to_instantaneous"
        )

    @staticmethod
    def _legend_positions(n, spacing):
        total_sp = (n - 1) * spacing
        sh = (1.0 - total_sp) / n
        return {
            i: {"x": 0.98, "y": 1.0 - (i - 1) * (sh + spacing) - 0.02}
            for i in range(1, n + 1)
        }

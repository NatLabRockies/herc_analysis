"""Phase 5 tests: the display package and the Scenario-backed time-series figure."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from herc_analysis import display
from herc_analysis.scenario import Scenario

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))
from conftest import _create_test_h5  # noqa: E402


@pytest.fixture(scope="module")
def scenario(tmp_path_factory) -> Scenario:
    path = tmp_path_factory.mktemp("display_fixture") / "hercules_output.h5"
    _create_test_h5(str(path))
    return Scenario(str(path))


def test_reexports_are_the_moved_objects():
    from herc_analysis.constants import SignalSubplot
    from herc_analysis.input_analysis import plot_histogram

    assert display.SignalSubplot is SignalSubplot
    assert display.plot_histogram is plot_histogram


def test_timeseries_figure_builds_plot_with_data(scenario):
    plotter = display.timeseries_figure(scenario)
    fig = plotter.plot_interactive(
        plot_dt=2, height_per_subplot=200, save_file=None, open_browser=False
    )
    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) > 0


def test_timeseries_figure_supports_derived_and_raw_signals(scenario):
    plotter = display.timeseries_figure(scenario)
    fig = plotter.plot_interactive(
        plot_dt=2,
        height_per_subplot=200,
        save_file=None,
        open_browser=False,
        signal_subplots=[
            display.SignalSubplot(
                columns="battery_soc", title="Battery SOC", y_label="SOC"
            ),
            display.SignalSubplot(
                columns="wind_farm.wind_speed_mean_background",
                title="Wind speed",
                y_label="m/s",
            ),
        ],
    )
    assert len(fig.data) > 0


def test_timeseries_figure_multi_scenario(scenario):
    plotter = display.timeseries_figure([scenario, scenario], scenario_names=["A", "B"])
    fig = plotter.plot_interactive(
        plot_dt=2, height_per_subplot=200, save_file=None, open_browser=False
    )
    assert len(fig.data) > 0


def test_legacy_plot_df_uses_old_names(scenario):
    from herc_analysis.display.timeseries_plot import _legacy_plot_df

    df = _legacy_plot_df(scenario)
    # old derived names present, kW power rescaled to MW
    assert "total_plant_power_mw" in df.columns
    assert "battery_power_mw" in df.columns
    assert "battery_soc" in df.columns
    # raw signal columns carried through
    assert "wind_farm.wind_speed_mean_background" in df.columns


def test_to_great_table_renders():
    df = pd.DataFrame({"case": ["a", "b"], "x": [1.0, 2.0]}).set_index("case")
    gt = display.to_great_table(df, title="T")
    assert gt is not None

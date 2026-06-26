"""Tests for the time-series figure (display.timeseries_figure over Scenario)."""

import sys
from pathlib import Path

import pytest

from herc_analysis import Scenario, SignalSubplot
from herc_analysis.display import timeseries_figure

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from conftest import _create_test_h5  # noqa: E402


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    path = tmp_path_factory.mktemp("plotting_fixture") / "hercules_output.h5"
    _create_test_h5(str(path))
    return Scenario(str(path))


def test_plot_single_scenario(scenario):
    plotter = timeseries_figure(scenario)
    fig = plotter.plot_interactive(
        plot_dt=2, height_per_subplot=200, save_file=None, open_browser=False
    )
    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) > 0


def test_plot_multiple_scenarios(scenario):
    plotter = timeseries_figure([scenario, scenario], scenario_names=["Run 1", "Run 2"])
    fig = plotter.plot_interactive(
        plot_dt=2, height_per_subplot=200, save_file=None, open_browser=False
    )
    assert fig is not None
    assert len(fig.data) > 2


def test_plot_with_signal_subplots(scenario):
    plotter = timeseries_figure(scenario)
    fig = plotter.plot_interactive(
        plot_dt=2,
        height_per_subplot=200,
        save_file=None,
        open_browser=False,
        signal_subplots=[
            SignalSubplot(
                columns="battery_soc",
                title="Battery SOC",
                y_label="SOC (%)",
                scale=100,
                reference_lines=[20, 80],
            ),
        ],
    )
    assert fig is not None
    assert len(fig.data) > 0


def test_plot_signal_subplot_multi_column(scenario):
    plotter = timeseries_figure(scenario)
    fig = plotter.plot_interactive(
        plot_dt=2,
        height_per_subplot=200,
        save_file=None,
        open_browser=False,
        signal_subplots=[
            SignalSubplot(
                columns=[
                    "wind_farm.wind_speed_mean_background",
                    "wind_farm.wind_speed_mean_withwakes",
                ],
                title="Wind Speed",
                y_label="Wind Speed (m/s)",
            ),
        ],
    )
    assert fig is not None
    assert len(fig.data) > 0


def test_component_power_subplot(scenario):
    plotter = timeseries_figure(scenario)
    ss = plotter.component_power_subplot("wind_farm")
    assert "wind_farm_power_mw" in (
        ss.columns if isinstance(ss.columns, list) else [ss.columns]
    )
    assert ss.y_label == "Power (MW)"


def test_print_available_signals(scenario, capsys):
    plotter = timeseries_figure(scenario)
    plotter.print_available_signals()
    captured = capsys.readouterr()
    assert "wind_farm" in captured.out
    assert "battery" in captured.out


def test_subplot_plan_structure(scenario):
    plotter = timeseries_figure(scenario)
    plan = plotter._build_subplot_plan(
        show_plant_power=True,
        show_market=True,
        signal_subplots=[SignalSubplot("battery_soc", title="SOC")],
    )
    ids = [s["id"] for s in plan]
    assert ids[0] == "signal_0"
    assert ids[1] == "plant_power"
    assert ids[2] == "market"


def test_subplot_plan_without_standard_plots(scenario):
    plotter = timeseries_figure(scenario)
    plan = plotter._build_subplot_plan(
        show_plant_power=False,
        show_market=False,
        signal_subplots=[SignalSubplot("battery_soc")],
    )
    ids = [s["id"] for s in plan]
    assert "plant_power" not in ids
    assert "market" not in ids
    assert len(ids) == 1

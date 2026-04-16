from herc_analysis import OutputAnalysis, PlotHerculesOutput, SignalSubplot


def test_plot_single_scenario():
    """Test creating a plot for a single scenario."""
    oa = OutputAnalysis("hercules_output.h5")
    plotter = PlotHerculesOutput(oa)
    fig = plotter.plot_interactive(plot_dt=120, height_per_subplot=200, save_file=None)
    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) > 0


def test_plot_multiple_scenarios():
    """Test creating a comparison plot for multiple scenarios."""
    oa1 = OutputAnalysis("hercules_output.h5")
    oa2 = OutputAnalysis("hercules_output.h5")
    plotter = PlotHerculesOutput([oa1, oa2], scenario_names=["Run 1", "Run 2"])
    fig = plotter.plot_interactive(plot_dt=120, height_per_subplot=200, save_file=None)
    assert fig is not None
    assert len(fig.data) > 2


def test_plot_plant_power_and_market_only():
    """Test showing only the two standard subplots."""
    oa = OutputAnalysis("hercules_output.h5")
    plotter = PlotHerculesOutput(oa)
    fig = plotter.plot_interactive(
        plot_dt=120,
        height_per_subplot=200,
        save_file=None,
    )
    assert fig is not None
    assert len(fig.data) > 0


def test_plot_with_signal_subplots():
    """Test adding custom signal subplots."""
    oa = OutputAnalysis("hercules_output.h5")
    plotter = PlotHerculesOutput(oa)
    fig = plotter.plot_interactive(
        plot_dt=120,
        height_per_subplot=200,
        save_file=None,
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


def test_plot_signal_subplot_multi_column():
    """Test a signal subplot with multiple columns."""
    oa = OutputAnalysis("hercules_output.h5")
    plotter = PlotHerculesOutput(oa)
    fig = plotter.plot_interactive(
        plot_dt=120,
        height_per_subplot=200,
        save_file=None,
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


def test_component_power_subplot():
    """Test the component_power_subplot convenience method."""
    oa = OutputAnalysis("hercules_output.h5")
    plotter = PlotHerculesOutput(oa)
    ss = plotter.component_power_subplot("wind_farm")
    assert "wind_farm_power_mw" in (
        ss.columns if isinstance(ss.columns, list) else [ss.columns]
    )
    assert ss.y_label == "Power (MW)"


def test_print_available_signals(capsys):
    """Test that print_available_signals produces output."""
    oa = OutputAnalysis("hercules_output.h5")
    plotter = PlotHerculesOutput(oa)
    plotter.print_available_signals()
    captured = capsys.readouterr()
    assert "wind_farm" in captured.out
    assert "battery" in captured.out


def test_subplot_plan_structure():
    """Test that _build_subplot_plan returns correct structure."""
    oa = OutputAnalysis("hercules_output.h5")
    plotter = PlotHerculesOutput(oa)
    plan = plotter._build_subplot_plan(
        show_plant_power=True,
        show_market=True,
        signal_subplots=[
            SignalSubplot("battery_soc", title="SOC"),
        ],
    )
    ids = [s["id"] for s in plan]
    assert ids[0] == "signal_0"
    assert ids[1] == "plant_power"
    assert ids[2] == "market"


def test_subplot_plan_without_standard_plots():
    """Test plan with standard plots disabled."""
    oa = OutputAnalysis("hercules_output.h5")
    plotter = PlotHerculesOutput(oa)
    plan = plotter._build_subplot_plan(
        show_plant_power=False,
        show_market=False,
        signal_subplots=[
            SignalSubplot("battery_soc"),
        ],
    )
    ids = [s["id"] for s in plan]
    assert "plant_power" not in ids
    assert "market" not in ids
    assert len(ids) == 1

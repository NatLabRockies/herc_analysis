"""Example demonstrating basic analysis of a single Hercules scenario.

Uses stored output from HERCULES Example 05 (wind farm + battery storage
with LMP-based control).
"""

from pathlib import Path

from herc_analysis import (
    OutputAnalysis,
    PlotHerculesOutput,
    SignalSubplot,
    TotalMetrics,
)

STORED_DATA = (
    Path(__file__).resolve().parents[2]
    / "stored_hercules_output"
    / "05_wind_and_storage_with_lmp"
    / "outputs"
    / "hercules_output.h5"
)


def main():
    oa = OutputAnalysis(str(STORED_DATA))

    print("\nDiscovered components:")
    for comp in oa.components:
        print(f"  {comp.name}: type={comp.component_type}, category={comp.category}")

    # Compute and display metrics
    tm = TotalMetrics(oa)
    tm.compute_metrics()
    tm.save_metrics("outputs/metrics.pkl")

    # Full interactive plot with wind speed, turbine powers, battery SOC,
    # and battery power subplots
    plotter = PlotHerculesOutput(oa)

    turbine_power_cols = oa.get_signal_columns("wind_farm", "turbine_powers")

    plotter.plot_interactive(
        save_file="outputs/single_scenario_full.html",
        plot_dt=10,
        signal_subplots=[
            SignalSubplot(
                columns="wind_farm.wind_speed_mean_background",
                title="Wind Speed",
                y_label="Wind Speed (m/s)",
            ),
            SignalSubplot(
                columns=turbine_power_cols[:3],
                title="Turbine Powers (first 3)",
                y_label="Power (kW)",
            ),
            SignalSubplot(
                columns="battery_soc",
                title="Battery State of Charge",
                y_label="SOC (%)",
                scale=100,
                reference_lines=[10, 90],
                reference_line_labels=["Min SOC", "Max SOC"],
            ),
            plotter.component_power_subplot("battery"),
        ],
    )

    # Plant power, battery power vs setpoint, and market
    plotter.plot_interactive(
        save_file="outputs/single_scenario_market.html",
        plot_dt=60,
        show_negative_ptc_line=False,
        shade_price_area=True,
        signal_subplots=[
            plotter.component_power_subplot("battery"),
        ],
    )


if __name__ == "__main__":
    main()

"""Example demonstrating comparison of two Hercules scenarios.

Compares stored output from:
  - HERCULES Example 02b (wind farm only, precomputed FLORIS)
  - HERCULES Example 05  (wind farm + battery storage with LMP)
"""

from pathlib import Path

from herc_analysis import OutputAnalysis, PlotHerculesOutput, TotalMetrics

STORED_DATA_DIR = Path(__file__).resolve().parents[2] / "stored_hercules_output"

WIND_ONLY_H5 = (
    STORED_DATA_DIR
    / "02b_wind_farm_realistic_inflow_precom_floris"
    / "outputs"
    / "hercules_output.h5"
)
WIND_STORAGE_H5 = (
    STORED_DATA_DIR / "05_wind_and_storage_with_lmp" / "outputs" / "hercules_output.h5"
)

SCENARIO_NAMES = ["Wind Only", "Wind + Storage"]


def main():
    oa_wind = OutputAnalysis(str(WIND_ONLY_H5))
    oa_storage = OutputAnalysis(str(WIND_STORAGE_H5))

    # Compute and compare metrics across scenarios
    tm = TotalMetrics(
        output_analysis=[oa_wind, oa_storage],
        scenario_names=SCENARIO_NAMES,
    )
    tm.compute_metrics()
    tm.compare_scenarios(include_pct_change=True)

    # Bar chart comparing total plant energy
    fig, ax = tm.plot_compare_scenarios(
        "plant.total_energy_mwh",
        plot_type="bar",
        color="steelblue",
    )
    fig.savefig("outputs/energy_comparison.png")

    # Multi-scenario interactive overlay
    plotter = PlotHerculesOutput(
        output_analysis=[oa_wind, oa_storage],
        scenario_names=SCENARIO_NAMES,
    )
    plotter.plot_interactive(
        plot_dt=10,
        signal_subplots=[
            plotter.component_power_subplot("wind_farm"),
        ],
        save_file="outputs/two_scenario_comparison.html",
    )


if __name__ == "__main__":
    main()

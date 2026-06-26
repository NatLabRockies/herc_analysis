"""Example demonstrating comparison of two Hercules scenarios.

Compares stored output from:
  - HERCULES Example 02b (wind farm only, precomputed FLORIS)
  - HERCULES Example 05  (wind farm + battery storage with LMP)

Ported to the refactored API: two ``Scenario`` objects feed the single
``Comparison`` engine; ``timeseries_figure`` draws the multi-scenario overlay.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from herc_analysis import Comparison, Scenario
from herc_analysis.display import timeseries_figure

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
    scenarios = [
        Scenario(str(WIND_ONLY_H5), name="Wind Only"),
        Scenario(str(WIND_STORAGE_H5), name="Wind + Storage"),
    ]

    # Compare metrics across scenarios with the single Comparison engine.
    cmp = Comparison.from_scenarios(scenarios)
    print("\nAnnual plant comparison:")
    print(cmp.table(["energy_mwh", "capacity_factor", "revenue_rt"], period="annual"))

    # Bar chart comparing total plant energy.
    fig, _ = cmp.plot("energy_mwh", period="total", color="steelblue")
    fig.savefig("outputs/energy_comparison.png")
    plt.close(fig)

    # Multi-scenario interactive overlay.
    plotter = timeseries_figure(scenarios, scenario_names=SCENARIO_NAMES)
    plotter.plot_interactive(
        plot_dt=10,
        signal_subplots=[
            plotter.component_power_subplot("wind_farm"),
        ],
        save_file="outputs/two_scenario_comparison.html",
    )


if __name__ == "__main__":
    main()

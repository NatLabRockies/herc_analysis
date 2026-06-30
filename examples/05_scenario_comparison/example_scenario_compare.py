"""Example: comparing metrics across scenarios with the Comparison engine.

Ported to the refactored API. The workflow is now:

1. Build a ``Scenario`` per case and write its long-format ``metrics.csv``.
2. Collect those files with the single :class:`~herc_analysis.Comparison` engine.
3. Build comparison tables in the per-year and cumulative views.
4. Render a ``great_tables`` table and a per-metric plot.

``Comparison`` replaces the old ``TotalMetrics`` list-mode and
``ScenarioComparison``; ``Scenario.metric_set`` writes the tidy file it reads.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from herc_analysis import Comparison, Scenario
from herc_analysis.display import to_great_table

EXAMPLE_DIR = Path(__file__).resolve().parent
STORED = EXAMPLE_DIR.parents[1] / "stored_hercules_output"
WORK_DIR = EXAMPLE_DIR / "outputs" / "cases"

# Map a friendly case name to its stored Hercules output.
SCENARIOS = {
    "wind_only": STORED / "02b_wind_farm_realistic_inflow_precom_floris",
    "wind_storage": STORED / "05_wind_and_storage_with_lmp",
}


def main():
    # Step 1: a Scenario per case, each writing its long-format metrics.csv.
    case_dirs = []
    for name, scenario_dir in SCENARIOS.items():
        scenario = Scenario(
            str(scenario_dir / "outputs" / "hercules_output.h5"), name=name
        )
        case_dir = WORK_DIR / name
        scenario.metric_set.to_csv(case_dir / "outputs" / "metrics.csv")
        case_dirs.append(case_dir)

    # Step 2: collect the per-case files into one comparison engine.
    cmp = Comparison.from_cases(case_dirs, case_names=list(SCENARIOS))

    # Step 3: comparison tables (per-year and cumulative views) at the plant level.
    annual = cmp.table(
        ["energy_mwh", "capacity_factor", "revenue_rt"],
        view="per_year",
        entity="plant",
    )
    print("\nAnnual plant comparison:")
    print(annual)

    total = cmp.table(["energy_mwh", "revenue_rt"], view="cumulative", entity="plant")
    print("\nTotal plant comparison:")
    print(total)

    # Per-component battery mileage (blank for the wind-only case).
    mileage = cmp.table("battery_mileage_soc", view="cumulative", entity="all")
    print("\nBattery mileage (total, per component):")
    print(mileage)

    Comparison.to_csv(annual, EXAMPLE_DIR / "outputs" / "annual_comparison.csv")

    # Step 4a: great_tables rendering (optional dependency).
    gt = to_great_table(
        annual,
        title="Scenario Comparison",
        subtitle="Plant metrics (annual)",
    )
    table_path = EXAMPLE_DIR / "outputs" / "annual_comparison.html"
    table_path.write_text(gt.as_raw_html())
    print("\nSaved great_tables table to outputs/annual_comparison.html")

    # Step 4b: plot a single metric across cases.
    cmp.plot("energy_mwh", view="cumulative", entity="plant")
    plt.savefig(EXAMPLE_DIR / "outputs" / "energy_total.png")
    print("Saved plot to outputs/energy_total.png")
    plt.show()


if __name__ == "__main__":
    main()

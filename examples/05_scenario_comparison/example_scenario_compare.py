"""Example: comparing metrics across scenarios with ScenarioComparison.

This example treats the two bundled stored simulations as two "cases" and
shows the full cross-scenario workflow:

1. Write a tidy per-case ``metrics.csv`` (the schema ScenarioComparison reads).
2. Collect those files with :class:`ScenarioComparison`.
3. Build annual / total comparison tables.
4. Render a ``great_tables`` table (if installed) and a per-metric plot.

In a real project the tidy ``metrics.csv`` is produced by a headless analysis
script and ScenarioComparison can run that script for you via
``ScenarioComparison.run_analysis_in_cases`` / ``from_cases(run_script=...)``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from herc_analysis import OutputAnalysis, ScenarioComparison, TotalMetrics



EXAMPLE_DIR = Path(__file__).resolve().parent
STORED = EXAMPLE_DIR.parents[1] / "stored_hercules_output"
WORK_DIR = EXAMPLE_DIR / "outputs" / "cases"

# Map a friendly case name to its stored Hercules output.
SCENARIOS = {
    "wind_only": STORED / "02b_wind_farm_realistic_inflow_precom_floris",
    "wind_storage": STORED / "05_wind_and_storage_with_lmp",
}

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def write_tidy_metrics(case_dir: Path, output_h5: Path) -> Path:
    """Compute metrics for one scenario and write the tidy metrics.csv.

    Args:
        case_dir (Path): Destination case directory (gets an ``outputs/``).
        output_h5 (Path): Path to the scenario's Hercules H5 output.

    Returns:
        Path: Path to the written ``metrics.csv``.
    """
    oa = OutputAnalysis(str(output_h5))
    metrics = TotalMetrics(oa).compute_metrics(display=False)
    sim_years = oa.total_simulation_time_s / SECONDS_PER_YEAR

    rows: list[dict] = [
        {
            "scope": "plant",
            "metric": "energy_mwh",
            "value": metrics["plant"]["total_energy_mwh"],
            "unit": "MWh",
            "scaling": "extensive",
        },
        {
            "scope": "plant",
            "metric": "capacity_factor",
            "value": metrics["plant"]["capacity_factor"],
            "unit": "-",
            "scaling": "intensive",
        },
        {
            "scope": "plant",
            "metric": "revenue_rt",
            "value": metrics["plant"]["total_revenue_rt_k"],
            "unit": "k$",
            "scaling": "extensive",
        },
    ]

    # Per-component energy and (for storage) battery mileage.
    for name, cm in metrics["components"].items():
        rows.append(
            {
                "scope": name,
                "metric": "energy_mwh",
                "value": cm["energy_mwh"],
                "unit": "MWh",
                "scaling": "extensive",
            }
        )
        if "battery_mileage_soc" in cm:
            rows.append(
                {
                    "scope": name,
                    "metric": "battery_mileage_soc",
                    "value": cm["battery_mileage_soc"],
                    "unit": "SOC",
                    "scaling": "extensive",
                }
            )

    df = pd.DataFrame(rows)
    df["sim_years"] = sim_years

    out = case_dir / "outputs" / "metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def main():
    # Step 1: produce a tidy metrics.csv for each case.
    case_dirs = []
    for name, scenario_dir in SCENARIOS.items():
        case_dir = WORK_DIR / name
        write_tidy_metrics(case_dir, scenario_dir / "outputs" / "hercules_output.h5")
        case_dirs.append(case_dir)

    # Step 2: collect the per-case files.
    sc = ScenarioComparison.from_cases(case_dirs, case_names=list(SCENARIOS))

    # Step 3: comparison tables (annual and total) at the plant level.
    annual = sc.compare(
        ["energy_mwh", "capacity_factor", "revenue_rt"],
        period="annual",
        scope="plant",
    )
    print("\nAnnual plant comparison:")
    print(annual)

    total = sc.compare(["energy_mwh", "revenue_rt"], period="total", scope="plant")
    print("\nTotal plant comparison:")
    print(total)

    # Per-component battery mileage (blank for the wind-only case).
    mileage = sc.compare("battery_mileage_soc", period="total", scope="all")
    print("\nBattery mileage (total, per component):")
    print(mileage)

    sc.to_csv(annual, EXAMPLE_DIR / "outputs" / "annual_comparison.csv")

    # Step 4a: great_tables rendering (optional dependency).

    gt = sc.to_great_table(
        annual,
        title="Scenario Comparison",
        subtitle="Plant metrics (annual)",
    )
    table_path = EXAMPLE_DIR / "outputs" / "annual_comparison.html"
    table_path.write_text(gt.as_raw_html())
    print("\nSaved great_tables table to outputs/annual_comparison.html")


    # Step 4b: plot a single metric across cases.
    sc.plot_metric("energy_mwh", period="total", scope="plant")
    plt.savefig(EXAMPLE_DIR / "outputs" / "energy_total.png")
    print("Saved plot to outputs/energy_total.png")
    plt.show()


if __name__ == "__main__":
    main()

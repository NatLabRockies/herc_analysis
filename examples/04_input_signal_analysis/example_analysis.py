"""Example 04 — Exploratory analysis of input signals.

Loads the MISO wind-farm Hercules output (which carries real-time and
day-ahead LMPs as external signals and spans roughly one year of UTC
time) and demonstrates the input-signal plot helpers in
:mod:`herc_analysis.display` (formerly ``herc_analysis.input_analysis``):

  * Donut plot of negative vs positive LMP, overall and by year.
  * Histograms of LMP and wind farm power.
  * Per-year boxplot of LMP.
  * Binned correlation of wind-farm power vs LMP.
  * Diurnal profile of LMP using local time.
  * Per-year summary statistics.

The h5 used here has no native wind-speed column, so the correlation plot
uses ``wind_farm_power_mw`` as the input signal.  Swap in a wind-speed or
irradiance column when running on a dataset that contains one.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from herc_analysis import (
    Scenario,
    plot_boxplot_by_year,
    plot_correlation,
    plot_diurnal,
    plot_histogram,
    plot_price_donut,
    summary_stats,
)
from herc_analysis.timeseries import add_local_time

# North Dakota wind farm (MISO Zone 1) — approximate coordinates.
LATITUDE = 47.0
LONGITUDE = -100.0

DATA_H5 = (
    Path(__file__).resolve().parents[1]
    / "02_miso_capacity_analysis_wind"
    / "hercules_output.h5"
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def main():
    scenario = Scenario(str(DATA_H5))
    ch = scenario.channels
    df = ch[["time_utc", "lmp_rt", "lmp_da"]].copy()
    # Channels keep power in kW; rescale to MW for this exploratory analysis.
    df["wind_farm_power_mw"] = ch["wind_farm__power_kw"] / 1000.0
    df = add_local_time(df, latitude=LATITUDE, longitude=LONGITUDE)

    # 1) Donut plots — overall and per-year.
    fig, _ = plot_price_donut(df, "lmp_rt", title="Real-time LMP")
    fig.savefig(OUTPUTS / "lmp_rt_donut.png", dpi=120, bbox_inches="tight")

    fig, _ = plot_price_donut(df, "lmp_rt", by_year=True, title="Real-time LMP by year")
    fig.savefig(OUTPUTS / "lmp_rt_donut_by_year.png", dpi=120, bbox_inches="tight")

    # 2) Histograms — LMP (default price bins) and wind power (auto bins).
    fig, _ = plot_histogram(df, "lmp_rt", xlabel="LMP RT ($/MWh)")
    fig.savefig(OUTPUTS / "lmp_rt_histogram.png", dpi=120, bbox_inches="tight")

    fig, _ = plot_histogram(df, "wind_farm_power_mw", xlabel="Wind farm power (MW)")
    fig.savefig(OUTPUTS / "wind_power_histogram.png", dpi=120, bbox_inches="tight")

    # 3) Boxplot of LMP by year.
    fig, _ = plot_boxplot_by_year(df, "lmp_rt", title="Real-time LMP by year")
    fig.savefig(OUTPUTS / "lmp_rt_boxplot_by_year.png", dpi=120, bbox_inches="tight")

    # 4) Correlation of wind output vs. real-time LMP.
    fig, _ = plot_correlation(
        df,
        x_col="wind_farm_power_mw",
        y_col="lmp_rt",
        kind="point",
        bin_width=5.0,
    )
    fig.savefig(
        OUTPUTS / "wind_power_vs_lmp_rt_point.png", dpi=120, bbox_inches="tight"
    )

    fig, _ = plot_correlation(
        df,
        x_col="wind_farm_power_mw",
        y_col="lmp_rt",
        kind="box",
        bin_width=5.0,
    )
    fig.savefig(OUTPUTS / "wind_power_vs_lmp_rt_box.png", dpi=120, bbox_inches="tight")

    # 5) Diurnal profile of real-time LMP (uses time_local).
    fig, _ = plot_diurnal(df, "lmp_rt", title="Real-time LMP — diurnal profile")
    fig.savefig(OUTPUTS / "lmp_rt_diurnal.png", dpi=120, bbox_inches="tight")

    # 6) Summary statistics — per-year, all signals.
    for col in ("lmp_rt", "lmp_da", "wind_farm_power_mw"):
        print(f"\nSummary stats for {col} (by year):")
        print(summary_stats(df, col, by_year=True).round(2))

    plt.show()

    plt.close("all")


if __name__ == "__main__":
    main()

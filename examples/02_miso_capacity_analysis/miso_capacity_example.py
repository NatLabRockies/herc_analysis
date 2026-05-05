from herc_analysis import OutputAnalysis
from herc_analysis.miso_capacity import compute_capacity_by_tier


def main():
    oa = OutputAnalysis("hercules_output.h5")

    df = oa.df

    print(df.columns)

    print(compute_capacity_by_tier(df, "wind_farm_power_mw", "north"))


if __name__ == "__main__":
    main()

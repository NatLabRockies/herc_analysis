import pandas as pd

from herc_analysis import OutputAnalysis
from herc_analysis.miso_capacity import compute_isac_dict


def main():
    oa = OutputAnalysis("hercules_output.h5")

    df = oa.df

    print(df["time_utc"].head())

    print(df["time_utc"].tail())

    print(df.columns)

    print(pd.DataFrame(compute_isac_dict(df, "wind_farm_power_mw", "north")))


if __name__ == "__main__":
    main()

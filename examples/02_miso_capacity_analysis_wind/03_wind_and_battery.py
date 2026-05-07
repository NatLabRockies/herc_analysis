# Add to the previous example a hypothetical battery of 20 MW, 4 hours
# That never discharges and so is always fully available at SOC = 1.0
# Compute the ISAC value for the wind and battery combined

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from herc_analysis import OutputAnalysis
from herc_analysis.miso_capacity import (
    compute_battery_availability,
    compute_isac_dict,
    get_acp,
    get_class_ucap_over_isac,
    get_days_per_season,
    limit_hourly_availability_contributions,
)

oa = OutputAnalysis("hercules_output.h5")

df = oa.df

df["battery_power"] = 20000.0 * np.ones(len(df))
df["battery_soc"] = 1.0 * np.ones(len(df))

interconnect_limit = 61500.0  # kW (61.5 MW)


# Compute the battery availability
df = compute_battery_availability(
    df,
    "battery",
    "battery_power",
    "battery_soc",
    battery_rated_power=20000.0,
    battery_rated_energy=80000.0,
    battery_min_soc=0.0,
    eta_discharge=1.0,
    return_df_hour=False,
)

# Pro-rata limit the power of wind and battery to the interconnect limit
df_limited_pro_rata = limit_hourly_availability_contributions(
    df,
    ["battery_availability", "wind_farm.power"],
    interconnect_limit,
    priority_order=False,
)

# Plot the original and limited power
fig, axarr = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
ax = axarr[0]
ax.plot(df["time_utc"], df["wind_farm.power"], label="Wind Power")
ax.plot(
    df["time_utc"], df_limited_pro_rata["wind_farm.power"], label="Wind Power Limited"
)
ax.legend()
ax.grid(True)

ax = axarr[1]
ax.plot(df["time_utc"], df["battery_availability"], label="Battery Availability")
ax.plot(
    df["time_utc"],
    df_limited_pro_rata["battery_availability"],
    label="Battery Availability Limited",
)
ax.legend()
ax.grid(True)

ax = axarr[2]
ax.plot(
    df["time_utc"],
    df["wind_farm.power"] + df["battery_availability"],
    label="Wind + Battery Power",
)
ax.plot(
    df["time_utc"],
    df_limited_pro_rata["wind_farm.power"]
    + df_limited_pro_rata["battery_availability"],
    label="Wind + Battery Power Limited",
)
ax.axhline(interconnect_limit, color="black", ls="--", label="Interconnect Limit")
ax.legend()
ax.grid(True)

# Compute the ISAC value for wind and battery seperately
isac_dict_wind = compute_isac_dict(df, "wind_farm.power", "north")
isac_dict_battery = compute_isac_dict(df, "battery_availability", "north")
print(isac_dict_wind)
print(isac_dict_battery)

# Combine the two dicts into a single dataframe with rows for each season
# and wind and battery ISAC values for columns
df_isac = pd.DataFrame(
    {
        "wind": {season: vals["ISAC"] for season, vals in isac_dict_wind.items()},
        "battery": {season: vals["ISAC"] for season, vals in isac_dict_battery.items()},
    }
)
df_isac.index.name = "season"

# Fill out a dataframe with the UCAP over ISAC for each season
# for wind and battery (battery uses the MISO "storage" resource class)
df_ucap_over_isac = pd.DataFrame(
    {
        "wind": {
            season: get_class_ucap_over_isac("wind", season) for season in df_isac.index
        },
        "battery": {
            season: get_class_ucap_over_isac("storage", season)
            for season in df_isac.index
        },
    }
)
df_ucap_over_isac.index.name = "season"

# SAC is ISAC * UCAP_Class / ISAC_Class
df_sac = df_isac * df_ucap_over_isac
df_sac.index.name = "season"

# Convert SAC to MW
df_sac = df_sac.mul(1 / 1000.0, axis=0)

print("ISAC:")
print(df_isac)
print("UCAP over ISAC:")
print(df_ucap_over_isac)
print("SAC:")
print(df_sac)


planning_year = 2026
zone = 1

# Days per season aligned to the SAC index
df_days_per_season = pd.DataFrame(
    {
        "days": [get_days_per_season(season, planning_year) for season in df_sac.index],
    },
    index=df_sac.index,
)

# ACP per season, broadcast across the same component columns as df_sac
acp_by_season = pd.Series(
    [get_acp(season, zone, planning_year) for season in df_sac.index],
    index=df_sac.index,
    name="acp_usd_per_mw_day",
)
df_acp = pd.DataFrame(
    dict.fromkeys(df_sac.columns, acp_by_season),
    index=df_sac.index,
)

# Revenue [$] = SAC [MW] * ACP [$/MW-day] * days [day]
df_revenue = df_sac.mul(df_acp).mul(df_days_per_season["days"], axis=0)
df_revenue.index.name = "season"

print("Days per season:")
print(df_days_per_season)
print("ACP ($/MW-day):")
print(df_acp)
print("Revenue ($):")
print(df_revenue)

plt.show()

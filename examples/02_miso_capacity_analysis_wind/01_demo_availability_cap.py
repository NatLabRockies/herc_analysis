# Demo of the availability cap functionality

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from herc_analysis.miso_capacity import limit_hourly_availability_contributions

n_steps = 1000
interconnect_limit = 10000.0  # kW
battery_power = np.linspace(0.0, 12000.0, n_steps)
wind_power = 8000.0 * np.ones(n_steps)

df = pd.DataFrame(
    {
        "time_utc": pd.date_range(
            start="2024-01-01 00:00:00", periods=n_steps, freq="15s", tz="UTC"
        ),
        "battery_power": battery_power,
        "wind_power": wind_power,
        "total_power": battery_power + wind_power,
    }
)

df_limited_pro_rata = limit_hourly_availability_contributions(
    df, ["battery_power", "wind_power"], interconnect_limit, priority_order=False
)
df_limited_priority_order = limit_hourly_availability_contributions(
    df, ["battery_power", "wind_power"], interconnect_limit, priority_order=True
)

df_limited_pro_rata["total_power"] = (
    df_limited_pro_rata["battery_power"] + df_limited_pro_rata["wind_power"]
)
df_limited_priority_order["total_power"] = (
    df_limited_priority_order["battery_power"] + df_limited_priority_order["wind_power"]
)

fig, axarr = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Pro-rate case
ax = axarr[0]
ax.plot(
    df["time_utc"],
    df_limited_pro_rata["total_power"],
    label="Total Power Limited",
    color="black",
    ls="-",
)
ax.plot(
    df["time_utc"], df["battery_power"], label="Battery Power", color="blue", ls="-"
)
ax.plot(
    df["time_utc"],
    df_limited_pro_rata["battery_power"],
    label="Battery Power Limited (Pro-Rata)",
    color="blue",
    ls="--",
)
ax.plot(df["time_utc"], df["wind_power"], label="Wind Power", color="red", ls="-")
ax.plot(
    df["time_utc"],
    df_limited_pro_rata["wind_power"],
    label="Wind Power Limited (Pro-Rata)",
    color="red",
    ls="--",
)
ax.axhline(interconnect_limit, color="black", ls="--", label="Interconnect Limit")

ax.legend()
ax.set_title("Pro-Rata Case")
ax.set_xlabel("Time")
ax.set_ylabel("Power (kW)")

# Priority order case
ax = axarr[1]
ax.plot(
    df["time_utc"],
    df_limited_priority_order["total_power"],
    label="Total Power Limited",
    color="black",
    ls="-",
)
ax.plot(
    df["time_utc"], df["battery_power"], label="Battery Power", color="blue", ls="-"
)
ax.plot(
    df["time_utc"],
    df_limited_priority_order["battery_power"],
    label="Battery Power Limited (Higher Priority)",
    color="blue",
    ls="--",
)
ax.plot(df["time_utc"], df["wind_power"], label="Wind Power", color="red", ls="-")
ax.plot(
    df["time_utc"],
    df_limited_priority_order["wind_power"],
    label="Wind Power Limited (Lower Priority)",
    color="red",
    ls="--",
)

ax.axhline(interconnect_limit, color="black", ls="--", label="Interconnect Limit")
ax.legend()
ax.set_title("Priority Order Case")
ax.set_xlabel("Time")

plt.show()

# Example demonstrating the battery availability computation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from herc_analysis.miso_capacity import compute_battery_availability

battery_rated_power = 10000.0  # kW (10 MW)
battery_rated_energy = 40000.0  # kWh (40 MWh, 4-hour storage)
battery_min_soc = 0.1
eta_discharge = 0.95  # discharge efficiency

dt = 10  # seconds
duration = 5 * 3600  # 5 hours
n_steps = int(duration / dt)
initial_soc = 1.0  # full charge

# Compute the power array assuming the power is rated_power for 5 hours
power = np.full(n_steps, battery_rated_power)

# Compute power internal, including efficiency
power_internal = power / eta_discharge

# Compute per step energy change
energy_change = power_internal * dt / 3600.0  # kWh

# Convert energy change to 0-1
energy_change_01 = energy_change / battery_rated_energy

# Compute the SOC array assuming the power is rated_power for 5 hours
soc = np.ones_like(power) * initial_soc
for i in range(n_steps):
    if i == 0:
        soc[i] = initial_soc
    else:
        soc[i] = soc[i - 1] - energy_change_01[i]
soc = np.clip(soc, battery_min_soc, 1.0)

# Reduce power when SOC is at or  below the minimum SOC
power = np.where(soc <= battery_min_soc, 0.0, power)


#  Build the dataframe
df = pd.DataFrame(
    {
        "time_utc": pd.date_range(
            start="2024-01-01 00:00:00", periods=n_steps, freq="10s", tz="UTC"
        ),
        "battery_power": power,
        "battery_soc": soc,
    }
)

# Add a column that is hours from the start of the simulation
df["hours_from_start"] = (
    df["time_utc"] - df["time_utc"].iloc[0]
).dt.total_seconds() / 3600.0

# Compute the battery availability
df, df_hourly = compute_battery_availability(
    df,
    "battery",
    "battery_power",
    "battery_soc",
    battery_rated_power,
    battery_rated_energy,
    battery_min_soc,
    eta_discharge,
    return_df_hour=True,
)

# Plot the results
fig, axarr = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Plot the battery power
ax = axarr[0]
ax.plot(
    df["hours_from_start"],
    df["battery_power"],
    label="Battery Power (Original)",
    color="black",
    ls="-",
)
ax.plot(
    df_hourly["hours_from_start"],
    df_hourly["power_hourly"],
    label="Battery Power (Hourly)",
    color="blue",
    ls="-",
)
ax.axhline(battery_rated_power, color="black", ls="--", label="Rated Power")
ax.set_ylabel("Power (kW)")
ax.legend()
ax.grid(True)

# Plot the battery SOC
ax = axarr[1]
ax.plot(
    df["hours_from_start"],
    df["battery_soc"],
    label="Battery SOC (Original)",
    color="black",
    ls="-",
)
ax.plot(
    df_hourly["hours_from_start"],
    df_hourly["soc_hour_start"],
    label="Battery SOC (Hourly Start)",
    color="green",
    ls="-",
)
ax.axhline(battery_min_soc, color="black", ls="--", label="Minimum SOC")
ax.set_ylabel("SOC (0-1)")
ax.legend()
ax.grid(True)

# Plot the battery availability
ax = axarr[2]

ax.plot(
    df["hours_from_start"],
    df["battery_availability"],
    label="Battery Availability",
    color="k",
    ls="-",
)
ax.plot(
    df_hourly["hours_from_start"],
    df_hourly["output_potential_power"],
    label="Battery Output Potential",
    color="r",
    ls="--",
)
ax.set_ylabel("Availability (kW)")
ax.legend()
ax.grid(True)

ax.set_xlabel("Hours from Start")
plt.show()

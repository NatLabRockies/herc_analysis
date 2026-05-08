"""Example 04 — Battery availability: adding a battery to the Hercules dataset.

Demonstrates compute_battery_availability using the Hercules simulation's time
axis.  A hypothetical 20 MW / 4-hour battery is added; two scenarios are shown
side by side to illustrate how SOC state drives the availability calculation:

  Scenario A — battery idle and fully charged (SOC = 1.0 always)
               → availability equals rated power at all times

  Scenario B — battery partially discharged (SOC decays from 0.4 to min_soc)
               → availability is energy-limited for the first few hours,
                 then drops to zero when SOC reaches min_soc

The key outputs from compute_battery_availability are:
  battery_availability  — max power the battery can deliver next hour [kW]
  grid_side_deliverable_energy — usable energy above min_soc × η [kWh]
  output_potential_power       — min(rated_power, deliverable_energy) [kW]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from herc_analysis import OutputAnalysis
from herc_analysis.miso_capacity import compute_battery_availability

DATA_DIR = Path(__file__).parent

BATTERY_RATED_POWER_KW = 20_000.0  # kW  (20 MW)
BATTERY_RATED_ENERGY_KWH = 80_000.0  # kWh (80 MWh — 4-hour duration)
BATTERY_MIN_SOC = 0.1
ETA_DISCHARGE = 0.95

oa = OutputAnalysis(DATA_DIR / "hercules_output.h5")
df_base = oa.df[["time_utc", "wind_farm.power"]].copy()

# ── Scenario A: idle battery, always fully charged ───────────────────────────
df_a = df_base.copy()
df_a["battery_power"] = 0.0
df_a["battery_soc"] = 1.0

df_a, df_a_hour = compute_battery_availability(
    df_a,
    component_name="battery",
    battery_power_column="battery_power",
    battery_soc_column="battery_soc",
    battery_rated_power=BATTERY_RATED_POWER_KW,
    battery_rated_energy=BATTERY_RATED_ENERGY_KWH,
    battery_min_soc=BATTERY_MIN_SOC,
    eta_discharge=ETA_DISCHARGE,
    return_df_hour=True,
)

# ── Scenario B: battery discharging — SOC decays from 40 % to min_soc ───────
# Simulate a fixed discharge: power = rated_power, SOC computed per time-step.
dt_s = 600.0  # 10-minute steps
energy_per_step_kwh = BATTERY_RATED_POWER_KW / ETA_DISCHARGE * dt_s / 3600.0
soc_change_per_step = energy_per_step_kwh / BATTERY_RATED_ENERGY_KWH

n = len(df_base)
soc_b = np.empty(n)
power_b = np.empty(n)
soc_b[0] = 0.40
for i in range(n):
    soc_b[i] = (
        max(BATTERY_MIN_SOC, soc_b[i - 1] - soc_change_per_step) if i > 0 else 0.40
    )
    power_b[i] = BATTERY_RATED_POWER_KW if soc_b[i] > BATTERY_MIN_SOC else 0.0

df_b = df_base.copy()
df_b["battery_power"] = power_b
df_b["battery_soc"] = soc_b

df_b, df_b_hour = compute_battery_availability(
    df_b,
    component_name="battery",
    battery_power_column="battery_power",
    battery_soc_column="battery_soc",
    battery_rated_power=BATTERY_RATED_POWER_KW,
    battery_rated_energy=BATTERY_RATED_ENERGY_KWH,
    battery_min_soc=BATTERY_MIN_SOC,
    eta_discharge=ETA_DISCHARGE,
    return_df_hour=True,
)

# Plot the first 8 hours (48 sub-hourly steps at 10-min resolution)
N_PLOT = 48

fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
fig.suptitle("Battery availability — compute_battery_availability demo", fontsize=13)

for col, (df, df_h, label) in enumerate(
    [
        (df_a, df_a_hour, "Scenario A: idle, full SOC"),
        (df_b, df_b_hour, "Scenario B: discharging from 40 % SOC"),
    ]
):
    t_sub = df["time_utc"].iloc[:N_PLOT]
    t_hr = df_h["time_utc"].iloc[:N_PLOT]

    ax = axes[0, col]
    ax.plot(
        t_sub,
        df["battery_soc"].iloc[:N_PLOT],
        color="steelblue",
        label="SOC (sub-hourly)",
    )
    ax.plot(
        t_hr,
        df_h["soc_hour_start"].iloc[:N_PLOT],
        "o--",
        color="steelblue",
        ms=4,
        label="SOC at hour start",
    )
    ax.axhline(
        BATTERY_MIN_SOC,
        color="red",
        ls="--",
        lw=1,
        label=f"Min SOC ({BATTERY_MIN_SOC})",
    )
    ax.set_title(label)
    ax.set_ylabel("SOC (0–1)")
    ax.legend(fontsize=8)
    ax.grid(True)

    ax = axes[1, col]
    ax.plot(
        t_sub,
        df["battery_power"].iloc[:N_PLOT] / 1000,
        color="darkorange",
        label="Power (sub-hourly)",
    )
    ax.plot(
        t_hr,
        df_h["power_hourly"].iloc[:N_PLOT] / 1000,
        "o--",
        color="darkorange",
        ms=4,
        label="Hourly mean",
    )
    ax.axhline(
        BATTERY_RATED_POWER_KW / 1000,
        color="green",
        ls="--",
        lw=1,
        label=f"Rated power ({BATTERY_RATED_POWER_KW / 1000:.0f} MW)",
    )
    ax.set_ylabel("Power (MW)")
    ax.legend(fontsize=8)
    ax.grid(True)

    ax = axes[2, col]
    ax.plot(
        t_sub,
        df["battery_availability"].iloc[:N_PLOT] / 1000,
        color="seagreen",
        lw=2,
        label="Availability (kW → MW)",
    )
    ax.plot(
        t_hr,
        df_h["output_potential_power"].iloc[:N_PLOT] / 1000,
        "o--",
        color="seagreen",
        ms=4,
        label="Output potential (hourly)",
    )
    ax.axhline(
        BATTERY_RATED_POWER_KW / 1000,
        color="green",
        ls="--",
        lw=1,
        label=f"Rated power ({BATTERY_RATED_POWER_KW / 1000:.0f} MW)",
    )
    ax.set_ylabel("Availability (MW)")
    ax.set_xlabel("Time (UTC)")
    ax.legend(fontsize=8)
    ax.grid(True)

plt.tight_layout()
plt.show()

print(
    f"Scenario A — mean battery availability: "
    f"{df_a['battery_availability'].mean() / 1000:.2f} MW"
)
print(
    f"Scenario B — mean battery availability: "
    f"{df_b['battery_availability'].mean() / 1000:.2f} MW"
)

"""L0 -- the Hercules boundary.

A thin adapter over :class:`hercules.hercules_output.HerculesOutput`. It hands
back a ``HerculesOutput`` (the single data owner) and a read-through scalar view
of its metadata; it copies no bulk data.

``RunMeta`` is the unit-normalization boundary: it converts Hercules' raw values
into named, unit-suffixed scalars once (``interconnect_limit`` kW -> ``interconnect_mw``,
``dt_log`` -> ``dt_s``) so the pure L1 functions receive clean scalars instead of
a heavyweight ``HerculesOutput``, and any Hercules attribute rename is absorbed
in this one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hercules.hercules_output import HerculesOutput


@dataclass(frozen=True)
class RunMeta:
    """Immutable scalar metadata for one run, lifted from ``HerculesOutput``.

    Only scalars live here so it is cheap to pass into pure functions. The
    dataframe and ``h_dict`` are NOT duplicated -- read them from the
    ``HerculesOutput`` the :class:`~herc_analysis.scenario.Scenario` holds.

    Attributes:
        dt_s (float): Logging timestep in seconds (``dt_log``).
        dt_sim_s (float): Simulation timestep in seconds.
        starttime_s (float): Simulation start time in seconds.
        endtime_s (float): Simulation end time in seconds.
        total_simulation_time_s (float): Total simulated duration in seconds.
        n_rows (int): Number of logged rows.
        interconnect_mw (float): Interconnect limit in MW.
    """

    dt_s: float
    dt_sim_s: float
    starttime_s: float
    endtime_s: float
    total_simulation_time_s: float
    n_rows: int
    interconnect_mw: float

    @classmethod
    def from_output(cls, ho: HerculesOutput) -> RunMeta:
        """Lift scalar metadata from a ``HerculesOutput``.

        Args:
            ho (HerculesOutput): The opened Hercules run.

        Returns:
            RunMeta: The normalized scalar metadata.
        """
        return cls(
            dt_s=float(ho.dt_log),
            dt_sim_s=float(ho.dt_sim),
            starttime_s=float(ho.starttime),
            endtime_s=float(ho.endtime),
            total_simulation_time_s=float(ho.total_simulation_time),
            n_rows=int(ho.total_rows_written),
            interconnect_mw=float(ho.h_dict["plant"]["interconnect_limit"]) / 1000.0,
        )

    @property
    def sim_hours(self) -> float:
        """Total simulated duration in hours."""
        return self.total_simulation_time_s / 3600.0

    @property
    def sim_years(self) -> float:
        """Total simulated duration in years (365.25-day years)."""
        return self.total_simulation_time_s / (365.25 * 24 * 3600)


def load_run(path: str | Path) -> HerculesOutput:
    """Open a Hercules H5 output. The single read entry point.

    Args:
        path (str | Path): Path to the Hercules output H5 file.

    Returns:
        HerculesOutput: The opened run.
    """
    return HerculesOutput(str(path))

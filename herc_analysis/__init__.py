"""herc_analysis -- analysis tools for Hercules hybrid-plant simulations.

Curated public API (post-refactor):

* ``Scenario``      -- analysis of one Hercules run (channels + metrics).
* ``MetricSet`` / ``MetricSpec`` -- the long-format metrics table.
* ``Comparison``    -- the single cross-scenario comparison engine.
* ``capacity``      -- capacity accreditation (``MisoCapacity``, availability
  providers); import from :mod:`herc_analysis.capacity`.
* ``display``       -- presentation (``timeseries_figure``, input plots, tables);
  import from :mod:`herc_analysis.display`.

The pre-refactor classes (``OutputAnalysis``, ``TotalMetrics``,
``ScenarioComparison``) have been removed -- see ``MIGRATION.md``.
"""

from importlib.metadata import version

__version__ = version("herc_analysis")

from .comparison import Comparison as Comparison
from .components import discover_components as discover_components
from .constants import COMPONENT_TYPE_TO_CATEGORY as COMPONENT_TYPE_TO_CATEGORY
from .constants import COMPONENT_TYPE_TO_MISO_CLASS as COMPONENT_TYPE_TO_MISO_CLASS
from .constants import PTC_PRICE as PTC_PRICE
from .constants import ComponentInfo as ComponentInfo
from .constants import SignalSubplot as SignalSubplot
from .display import plot_boxplot_by_year as plot_boxplot_by_year
from .display import plot_correlation as plot_correlation
from .display import plot_diurnal as plot_diurnal
from .display import plot_histogram as plot_histogram
from .display import plot_price_donut as plot_price_donut
from .display import summary_stats as summary_stats
from .display import timeseries_figure as timeseries_figure
from .display import to_great_table as to_great_table
from .io import RunMeta as RunMeta
from .io import load_run as load_run
from .metrics import MetricSet as MetricSet
from .metrics import MetricSpec as MetricSpec
from .scenario import Scenario as Scenario

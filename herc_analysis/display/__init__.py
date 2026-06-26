"""L6 -- presentation package.

Pure presentation: the Plotly time-series figure, great_tables rendering,
input-signal plots, the color palette and the ``SignalSubplot`` spec. No metric
or capacity logic lives here.
"""

from __future__ import annotations

from herc_analysis.display.colors import (
    INFRASTRUCTURE_COLORS as INFRASTRUCTURE_COLORS,
)
from herc_analysis.display.colors import (
    get_component_color as get_component_color,
)
from herc_analysis.display.colors import (
    get_detail_color as get_detail_color,
)
from herc_analysis.display.input_plots import (
    plot_boxplot_by_year as plot_boxplot_by_year,
)
from herc_analysis.display.input_plots import (
    plot_correlation as plot_correlation,
)
from herc_analysis.display.input_plots import (
    plot_diurnal as plot_diurnal,
)
from herc_analysis.display.input_plots import (
    plot_histogram as plot_histogram,
)
from herc_analysis.display.input_plots import (
    plot_price_donut as plot_price_donut,
)
from herc_analysis.display.input_plots import (
    summary_stats as summary_stats,
)
from herc_analysis.display.subplots import SignalSubplot as SignalSubplot
from herc_analysis.display.tables import to_great_table as to_great_table
from herc_analysis.display.timeseries_plot import (
    PlotHerculesOutput as PlotHerculesOutput,
)
from herc_analysis.display.timeseries_plot import (
    timeseries_figure as timeseries_figure,
)

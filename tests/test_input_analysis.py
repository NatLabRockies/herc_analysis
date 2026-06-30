"""Tests for the input-signal plot helpers (herc_analysis.display.input_plots)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from herc_analysis import (
    plot_boxplot_by_year,
    plot_correlation,
    plot_diurnal,
    plot_histogram,
    plot_price_donut,
    summary_stats,
)


@pytest.fixture
def df_inputs():
    """Synthetic two-year DataFrame with prices and a correlated wind speed."""
    rng = np.random.default_rng(seed=42)
    n = 24 * 365 * 2  # 2 years, hourly
    time_utc = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    wind_speed = np.clip(rng.normal(8.0, 3.0, n), 0.0, None)
    # Prices anti-correlated with wind: higher wind -> lower price + noise.
    lmp_rt = 60.0 - 4.0 * wind_speed + rng.normal(0.0, 15.0, n)
    lmp_da = 55.0 - 3.5 * wind_speed + rng.normal(0.0, 10.0, n)
    return pd.DataFrame(
        {
            "time_utc": time_utc,
            "time_local": time_utc.tz_convert("US/Central"),
            "wind_speed": wind_speed,
            "lmp_rt": lmp_rt,
            "lmp_da": lmp_da,
        }
    )


@pytest.fixture(autouse=True)
def _close_figs():
    """Close any open matplotlib figures after each test."""
    yield
    plt.close("all")


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_missing_time_utc_raises():
    df = pd.DataFrame({"lmp_rt": [1.0, -2.0, 3.0]})
    with pytest.raises(ValueError, match="time_utc"):
        plot_price_donut(df)


def test_missing_value_col_raises(df_inputs):
    with pytest.raises(ValueError, match="not in DataFrame"):
        plot_histogram(df_inputs, "does_not_exist")


# ----------------------------------------------------------------------
# Donut
# ----------------------------------------------------------------------


def test_plot_price_donut_single(df_inputs):
    fig, ax = plot_price_donut(df_inputs, "lmp_rt")
    assert isinstance(fig, Figure)
    # Pie has 2 wedge patches plus the centre white circle.
    assert any(p.get_label() == "Negative" for p in ax.patches[:2])


def test_plot_price_donut_by_year(df_inputs):
    fig, axes = plot_price_donut(df_inputs, "lmp_rt", by_year=True)
    assert isinstance(fig, Figure)
    assert len(axes) == 2  # 2023 and 2024


# ----------------------------------------------------------------------
# Histogram
# ----------------------------------------------------------------------


def test_plot_histogram_price_default_bins(df_inputs):
    fig, ax = plot_histogram(df_inputs, "lmp_rt")
    assert isinstance(fig, Figure)
    assert ax.get_xlabel() == "lmp_rt"


def test_plot_histogram_wind_speed_auto_bins(df_inputs):
    fig, ax = plot_histogram(df_inputs, "wind_speed", xlabel="Wind Speed (m/s)")
    assert isinstance(fig, Figure)
    assert ax.get_xlabel() == "Wind Speed (m/s)"


def test_plot_histogram_by_year(df_inputs):
    fig, ax = plot_histogram(df_inputs, "lmp_rt", by_year=True)
    assert isinstance(fig, Figure)


# ----------------------------------------------------------------------
# Boxplot
# ----------------------------------------------------------------------


def test_plot_boxplot_by_year(df_inputs):
    fig, ax = plot_boxplot_by_year(df_inputs, "lmp_rt")
    assert isinstance(fig, Figure)
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "2023" in tick_labels and "2024" in tick_labels


# ----------------------------------------------------------------------
# Correlation
# ----------------------------------------------------------------------


def test_plot_correlation_point(df_inputs):
    fig, ax = plot_correlation(df_inputs, "wind_speed", "lmp_rt", kind="point")
    assert isinstance(fig, Figure)
    # Pearson r reported in title and is negative for our synthetic data.
    assert "Pearson" in ax.get_title()


def test_plot_correlation_box(df_inputs):
    fig, ax = plot_correlation(df_inputs, "wind_speed", "lmp_rt", kind="box")
    assert isinstance(fig, Figure)


def test_plot_correlation_invalid_kind(df_inputs):
    with pytest.raises(ValueError, match="kind must"):
        plot_correlation(df_inputs, "wind_speed", "lmp_rt", kind="scatter")


# ----------------------------------------------------------------------
# Diurnal
# ----------------------------------------------------------------------


def test_plot_diurnal_uses_time_local(df_inputs):
    fig, ax = plot_diurnal(df_inputs, "lmp_rt")
    assert isinstance(fig, Figure)
    # 24 hour-of-day values on the central line.
    line = ax.get_lines()[0]
    assert len(line.get_xdata()) == 24


def test_plot_diurnal_falls_back_to_utc(df_inputs):
    df = df_inputs.drop(columns=["time_local"])
    with pytest.warns(UserWarning, match="falling back"):
        fig, _ax = plot_diurnal(df, "lmp_rt")
    assert isinstance(fig, Figure)


# ----------------------------------------------------------------------
# Summary stats
# ----------------------------------------------------------------------


def test_summary_stats_basic(df_inputs):
    result = summary_stats(df_inputs, "lmp_rt")
    assert list(result.index) == ["all"]
    assert {
        "count",
        "mean",
        "std",
        "min",
        "median",
        "max",
        "pct_negative",
    } <= set(result.columns)
    assert result.loc["all", "count"] == len(df_inputs)


def test_summary_stats_by_year(df_inputs):
    result = summary_stats(df_inputs, "lmp_rt", by_year=True)
    assert list(result.index) == [2023, 2024]
    assert result["count"].sum() == len(df_inputs)


def test_summary_stats_pct_negative():
    times = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({"time_utc": times, "lmp": [-1.0, -2.0, 3.0] + [1.0] * 7})
    result = summary_stats(df, "lmp")
    assert result.loc["all", "pct_negative"] == pytest.approx(20.0)

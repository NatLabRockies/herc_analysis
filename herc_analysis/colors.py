"""Color palettes and helpers for herc_analysis plotting."""

CATEGORY_PALETTES: dict[str, list[str]] = {
    "generator": [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#d62728",  # red
        "#17becf",  # cyan
        "#bcbd22",  # olive
    ],
    "storage": [
        "#9467bd",  # purple
        "#7b4397",  # dark purple
        "#c5b0d5",  # light purple
    ],
    "load": [
        "#2ca02c",  # green
        "#32cd32",  # lime
        "#98df8a",  # light green
    ],
}

INFRASTRUCTURE_COLORS = {
    "total_power": "#000000",
    "interconnect": "#dc143c",
    "reference_line": "#808080",
    "gray_light": "#d3d3d3",
    "ptc_revenue": "#2ca02c",
    "market_rt": "#2f2f2f",
    "market_da": "#696969",
}

# Turbine-level detail colours (cycled for array signals)
DETAIL_PALETTES: dict[str, list[str]] = {
    "turbine_powers": [
        "#4682b4",
        "#6495ed",
        "#1e90ff",
        "#0077be",
        "#5f9ea0",
        "#00bfff",
    ],
    "wind_speed": [
        "#5da5db",  # background
        "#0d5aa7",  # waked
    ],
    "irradiance": [
        "#ffa500",
        "#ff8c00",
    ],
}


def get_component_color(category: str, index: int = 0) -> str:
    """Return a colour for a component given its category and instance index.

    Args:
        category (str): 'generator', 'storage', or 'load'.
        index (int, optional): Instance index within category. Defaults to 0.

    Returns:
        str: Hex colour string.
    """
    palette = CATEGORY_PALETTES.get(category, CATEGORY_PALETTES["generator"])
    return palette[index % len(palette)]


def get_detail_color(signal: str, index: int = 0) -> str:
    """Return a colour for a detail signal subplot.

    Args:
        signal (str): Signal family name (e.g. 'turbine_powers', 'wind_speed').
        index (int, optional): Sub-index within the signal. Defaults to 0.

    Returns:
        str: Hex colour string.
    """
    palette = DETAIL_PALETTES.get(signal, CATEGORY_PALETTES["generator"])
    return palette[index % len(palette)]

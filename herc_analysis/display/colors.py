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

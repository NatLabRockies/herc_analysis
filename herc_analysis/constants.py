"""Constants and component registry for herc_analysis."""

from dataclasses import dataclass

from hercules.hybrid_plant import COMPONENT_REGISTRY

PTC_PRICE = 26.00

COMPONENT_TYPE_TO_CATEGORY: dict[str, str] = {
    ctype: cls.component_category for ctype, cls in COMPONENT_REGISTRY.items()
}

VALID_CATEGORIES = frozenset({"generator", "load", "storage"})


@dataclass(frozen=True)
class ComponentInfo:
    """Metadata for a single plant component.

    Args:
        name (str): User-chosen component name (h_dict key).
        component_type (str): Hercules class name string.
        category (str): One of 'generator', 'load', or 'storage'.
    """

    name: str
    component_type: str
    category: str


@dataclass
class SignalSubplot:
    """Specification for a generic signal subplot.

    Args:
        columns (str | list[str]): DataFrame column name(s) to plot.
        title (str, optional): Subplot title. Auto-generated from columns
            if empty. Defaults to "".
        y_label (str, optional): Y-axis label. Defaults to "".
        scale (float, optional): Multiply y-values by this factor.
            Defaults to 1.0.
        reference_lines (list[float], optional): Horizontal reference lines
            to draw. Defaults to None.
        reference_line_labels (list[str], optional): Labels for the
            reference lines. Must match the length of reference_lines
            when provided. Defaults to None.
    """

    columns: str | list[str]
    title: str = ""
    y_label: str = ""
    scale: float = 1.0
    reference_lines: list[float] | None = None
    reference_line_labels: list[str] | None = None

"""L1 -- component records, discovery, and registries.

``ComponentInfo`` and the category / MISO-class registries currently live in
:mod:`herc_analysis.constants` and are re-exported here so callers can import
them from their eventual home. ``discover_components`` is the pure discovery
routine lifted out of ``OutputAnalysis._discover_components``.
"""

from __future__ import annotations

from herc_analysis.constants import (
    COMPONENT_TYPE_TO_CATEGORY as COMPONENT_TYPE_TO_CATEGORY,
)
from herc_analysis.constants import (
    COMPONENT_TYPE_TO_MISO_CLASS as COMPONENT_TYPE_TO_MISO_CLASS,
)
from herc_analysis.constants import (
    ComponentInfo as ComponentInfo,
)


def discover_components(h_dict: dict) -> list[ComponentInfo]:
    """Discover plant components from an ``h_dict`` by component type.

    Reproduces ``OutputAnalysis._discover_components``: iterate the dict's
    sub-dicts, read ``component_type``, map it to a category via
    :data:`COMPONENT_TYPE_TO_CATEGORY`, and skip entries with no type or an
    unknown type. Unlike the original this routine is silent (no prints), which
    does not affect any numeric output.

    Args:
        h_dict (dict): The Hercules run dictionary.

    Returns:
        list[ComponentInfo]: Discovered components in ``h_dict`` order.
    """
    components: list[ComponentInfo] = []
    for key, val in h_dict.items():
        if not isinstance(val, dict):
            continue
        component_type = val.get("component_type", "")
        if not component_type:
            continue
        category = COMPONENT_TYPE_TO_CATEGORY.get(component_type)
        if category is None:
            continue
        components.append(ComponentInfo(key, component_type, category))
    return components

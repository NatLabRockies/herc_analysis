import pandas as pd
from hercules.hybrid_plant import COMPONENT_REGISTRY

from herc_analysis.capacity._miso_engine import RESOURCE_CLASS_UCAP_CSV_PATH
from herc_analysis.constants import (
    COMPONENT_TYPE_TO_CATEGORY,
    COMPONENT_TYPE_TO_MISO_CLASS,
)


def test_miso_class_map_keys_are_registry_types():
    registry_types = set(COMPONENT_REGISTRY.keys())
    assert set(COMPONENT_TYPE_TO_MISO_CLASS).issubset(registry_types)


def test_miso_class_map_values_are_valid_resource_classes():
    ucap_df = pd.read_csv(RESOURCE_CLASS_UCAP_CSV_PATH)
    valid_classes = set(ucap_df["resource_class"].astype(str).str.strip())
    assert set(COMPONENT_TYPE_TO_MISO_CLASS.values()).issubset(valid_classes)


def test_miso_class_map_excludes_loads():
    load_types = [
        ctype
        for ctype, category in COMPONENT_TYPE_TO_CATEGORY.items()
        if category == "load"
    ]
    for ctype in load_types:
        assert ctype not in COMPONENT_TYPE_TO_MISO_CLASS


def test_battery_types_map_to_storage_class():
    assert COMPONENT_TYPE_TO_MISO_CLASS["BatterySimple"] == "storage"
    assert COMPONENT_TYPE_TO_MISO_CLASS["BatteryLithiumIon"] == "storage"


def test_solar_type_maps_to_solar_class():
    assert COMPONENT_TYPE_TO_MISO_CLASS["SolarPySAMPVWatts"] == "solar"

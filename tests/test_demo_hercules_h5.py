import json

import h5py


def test_demo_file_has_component_type():
    """Test that test H5 always includes component_type in h_dict."""
    with h5py.File("demo_test.h5", "r") as hf:
        h_dict = json.loads(hf["metadata"].attrs["h_dict"])

    assert h_dict["wind_farm"]["component_type"] == "WindFarm"
    assert h_dict["solar_farm"]["component_type"] == "SolarPySAMPVWatts"
    assert h_dict["battery"]["component_type"] == "BatterySimple"


def test_demo_file_has_expected_datasets():
    """Test that test H5 contains expected data groups."""
    with h5py.File("demo_test.h5", "r") as hf:
        assert "data" in hf
        assert "time" in hf["data"]
        assert "components" in hf["data"]
        assert "wind_farm.power" in hf["data/components"]
        assert "solar_farm.power" in hf["data/components"]
        assert "battery.power" in hf["data/components"]

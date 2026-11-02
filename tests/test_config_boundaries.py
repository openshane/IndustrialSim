import pytest

from industrial_sim.client import IndustrialSimClient
from industrial_sim.core.config import ConfigurationError, CycleSpec, DeviceSpec
from industrial_sim.core.errors import DeviceError, FAILURE


@pytest.mark.parametrize("construct,field", [
    (lambda: CycleSpec(duration_ms=10**500), "cycle.duration_ms"),
    (lambda: CycleSpec(points=[{"pose": [10**500, 0, 0, 0, 0, 0]}]), "cycle.points"),
    (lambda: DeviceSpec("arm", "duco", {"home_pose": [10**500, 0, 0, 0, 0, 0]}), "home_pose"),
])
def test_unrepresentable_numbers_have_configuration_field_errors(construct, field):
    with pytest.raises(ConfigurationError, match=field):
        construct()


@pytest.mark.parametrize("response", [{}, {"configured": 1}, {"configured": True, "config": None}])
def test_invalid_remote_configuration_is_a_device_error(monkeypatch, response):
    client = IndustrialSimClient("http://127.0.0.1:12345")
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: response)
    with pytest.raises(DeviceError) as error:
        client.configuration()
    assert error.value.code == FAILURE

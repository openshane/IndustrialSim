from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from industrial_sim.core.errors import CANCELLED, DeviceError
from industrial_sim.devices.robot import VirtualRobot


def test_stop_cancels_a_request_still_validating_its_path(monkeypatch):
    robot = VirtualRobot()
    robot.init()
    entered = threading.Event()
    release = threading.Event()
    original = robot._point

    def delayed_validation(point):
        entered.set()
        assert release.wait(2)
        return original(point)

    monkeypatch.setattr(robot, "_point", delayed_validation)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(robot.move_path, [{"pose": [0] * 6}])
        try:
            assert entered.wait(1)
            robot.stop()
        finally:
            release.set()
        with pytest.raises(DeviceError) as failure:
            pending.result(timeout=2)
    assert failure.value.code == CANCELLED
    assert robot.get_state()["status"] != "running"

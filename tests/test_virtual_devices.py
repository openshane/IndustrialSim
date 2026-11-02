"""无网络的设备状态机回归；不代表真实机器人或厂商协议验收。"""

import threading
import time

import pytest

from industrial_sim.core.errors import (
    BUSY,
    CANCELLED,
    FAULT,
    INVALID_ARGUMENT,
    NOT_CONNECTED,
    UNSUPPORTED,
    DeviceError,
)
from industrial_sim.devices.io import VirtualIO
from industrial_sim.devices.robot import VirtualRobot
from industrial_sim.devices.warning_light import VirtualWarningLight


@pytest.fixture
def virtual_io():
    device = VirtualIO()
    device.init()
    yield device
    device.uninit()


def begin_io_wait(device, monkeypatch, **args):
    waiting = threading.Event()
    done = threading.Event()
    outcome = {}
    original_wait = device._condition.wait

    def observe_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    def run():
        try:
            outcome["value"] = device.wait(**args)
        except Exception as error:
            outcome["error"] = error
        finally:
            done.set()

    monkeypatch.setattr(device._condition, "wait", observe_wait)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert waiting.wait(1.0), "ReadWait returned a level instead of arming for an edge"
    return done, outcome, thread


def test_io_wait_requires_new_matching_edge_including_falling(virtual_io, monkeypatch):
    virtual_io.inject_input(0, 1)
    done, outcome, thread = begin_io_wait(
        virtual_io, monkeypatch, index=0, timeout_ms=1000
    )
    virtual_io.inject_input(0, 1)
    virtual_io.inject_input(1, 1)
    assert not done.wait(0.03), "same level or another bit incorrectly completed ReadWait"
    edge = virtual_io.inject_input(0, 0)
    assert done.wait(1.0)
    thread.join()
    assert outcome["value"] == {"value": 0, "sequence": edge["sequence"]}


def test_io_cursor_preserves_edge_that_arrives_before_wait_call(virtual_io):
    cursor = virtual_io.get_state()["sequence"]
    first = virtual_io.inject_input(3, 1)
    virtual_io.inject_input(3, 0)
    assert virtual_io.wait(index=3, after_sequence=cursor, timeout_ms=10) == first
    assert virtual_io.wait(index=3, after_sequence=first["sequence"], timeout_ms=10)["value"] == 0


def test_io_register_wait_ignores_other_port_and_returns_byte(virtual_io, monkeypatch):
    virtual_io.inject_input(2, 1)
    done, outcome, thread = begin_io_wait(
        virtual_io, monkeypatch, offset=0, timeout_ms=1000
    )
    virtual_io.inject_input(9, 1)
    assert not done.wait(0.03)
    edge = virtual_io.inject_input(2, 0)
    assert done.wait(1.0)
    thread.join()
    assert outcome["value"] == {"value": 0, "sequence": edge["sequence"]}
    assert virtual_io.read_reg(1)["value"] == 2


def test_io_timeout_uses_milliseconds(virtual_io):
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        virtual_io.wait(index=0, timeout_ms=25)
    elapsed = time.monotonic() - started
    assert 0.015 <= elapsed < 0.5


def test_io_uninit_and_immediate_init_cancel_old_wait(virtual_io, monkeypatch):
    virtual_io.set(0, 1)
    done, outcome, thread = begin_io_wait(virtual_io, monkeypatch, index=0, timeout_ms=0)
    # Keep the state lock across both transitions so the waiter only observes
    # the new connected generation, rather than relying on a scheduling gap.
    with virtual_io._condition:
        virtual_io.uninit()
        virtual_io.init()
    assert done.wait(1.0)
    thread.join()
    assert isinstance(outcome["error"], DeviceError)
    assert outcome["error"].code == CANCELLED
    assert virtual_io.get_state()["outputs"] == [0] * 16
    assert virtual_io.get_state()["connected"]


def test_io_expired_cursor_fails_instead_of_silently_losing_edges(virtual_io):
    for index in range(4097):
        virtual_io.inject_input(0, (index + 1) % 2)
    with pytest.raises(DeviceError) as caught:
        virtual_io.wait(index=0, after_sequence=0, timeout_ms=1)
    assert caught.value.code == INVALID_ARGUMENT


@pytest.fixture
def robot():
    device = VirtualRobot("flexiv")
    device.init()
    yield device
    device.uninit()


@pytest.mark.parametrize("brand,dof,frame", [("flexiv", 7, "base"), ("duco", 6, "world"), ("kuka", 6, "world")])
def test_robot_profiles_accept_joint_motion_but_submission_is_not_completion(brand, dof, frame):
    device = VirtualRobot(brand)
    device.init()
    accepted = device.move_joint([20] * dof, duration_ms=200)
    state = device.get_state()
    assert state["motion_id"] == accepted["motion_id"]
    assert state["status"] == "running"
    assert state["progress"] == 0
    assert not state["finished"]
    assert state["pose_frame"] == frame
    device.advance(0.05)
    assert device.get_state()["joints"] == pytest.approx([5] * dof)
    assert device.get_state()["progress"] == pytest.approx(0.25)
    device.advance(0.15)
    assert device.get_state()["joints"] == pytest.approx([20] * dof)
    assert device.get_state()["finished"]
    device.uninit()


def test_robot_path_progress_pause_resume_and_completion(robot):
    robot.move_path([{"pose": [100, 0, 0, 0, 0, 0]}, {"pose": [200, 0, 0, 0, 0, 0]}], duration_ms=1000)
    robot.advance(0.25)
    state = robot.get_state()
    assert state["pose"][0] == pytest.approx(50)
    assert state["waypoint_index"] == 0
    robot.pause()
    paused = robot.get_state()
    robot.advance(10)
    assert robot.get_state() == paused
    robot.resume()
    robot.advance(0.5)
    state = robot.get_state()
    assert state["pose"][0] == pytest.approx(150)
    assert state["waypoint_index"] == 1
    assert state["progress"] == pytest.approx(0.75)
    robot.advance(0.25)
    state = robot.get_state()
    assert state["status"] == "completed"
    assert state["finished"] and not state["moving"]
    assert state["progress"] == 1
    assert state["pose"][0] == 200


@pytest.mark.parametrize(
    "points",
    [
        [],
        [{}],
        [{"pose": [0] * 5}],
        [{"pose": [float("nan"), 0, 0, 0, 0, 0]}],
        [{"pose": [float("inf"), 0, 0, 0, 0, 0]}],
        [{"pose": [True, 0, 0, 0, 0, 0]}],
        [{"joints": [0] * 6}],
        [{"pose": [0] * 6, "process": None}],
        [{"pose": [0] * 6, "is_contact": 1}],
        [{"pose": [0] * 6, "process": {"valid": True, "tcp_vel": 0, "tcp_acc": 1}}],
        [{"pose": [0] * 6, "is_contact": True, "process": {"valid": False, "force": ["bad", 0, 0]}}],
        [{"pose": [0] * 6, "is_contact": True, "process": {"force": [float("nan"), 0, 0]}}],
        [{"pose": [0] * 6}, {"pose": [0] * 5}],
    ],
)
def test_invalid_robot_path_is_rejected_without_side_effects(robot, points):
    before = robot.get_state()
    with pytest.raises(DeviceError) as caught:
        robot.move_path(points, duration_ms=100)
    assert caught.value.code == INVALID_ARGUMENT
    assert robot.get_state() == before
    assert robot.get_path() == []


@pytest.mark.parametrize("extra", [{"reference_frame": "workpiece"}, {"ext_axes": [1]}])
def test_unsupported_path_features_are_not_silently_accepted(robot, extra):
    before = robot.get_state()
    with pytest.raises(DeviceError) as caught:
        robot.move_path([{"pose": [0] * 6, **extra}])
    assert caught.value.code == UNSUPPORTED
    assert robot.get_state() == before


@pytest.mark.parametrize("duration", [0, -1, True, float("inf"), float("nan")])
def test_invalid_duration_does_not_allocate_a_motion(robot, duration):
    before = robot.get_state()
    with pytest.raises(DeviceError) as caught:
        robot.move_path([{"pose": [0] * 6}], duration_ms=duration)
    assert caught.value.code == INVALID_ARGUMENT
    assert robot.get_state() == before


def test_active_motion_rejects_replacement_and_stale_stop(robot):
    first = robot.move_joint([10] * 7, duration_ms=1000)["motion_id"]
    robot.advance(0.2)
    before = robot.get_state()
    with pytest.raises(DeviceError) as caught:
        robot.move_joint([20] * 7)
    assert caught.value.code == BUSY
    assert robot.get_state() == before
    robot.stop(first)
    cancelled = robot.get_state()
    robot.advance(2)
    assert robot.get_state() == cancelled
    second = robot.move_joint([30] * 7, duration_ms=1000)["motion_id"]
    assert second > first
    with pytest.raises(DeviceError) as caught:
        robot.stop(first)
    assert caught.value.code == INVALID_ARGUMENT
    assert robot.get_state()["motion_id"] == second
    assert robot.get_state()["status"] == "running"


def test_fault_blocks_motion_and_clear_allows_a_new_motion(robot):
    robot.move_path([{"pose": [10, 0, 0, 0, 0, 0], "is_contact": True, "process": {"force": [1, 2, 3]}}], duration_ms=1000)
    robot.advance(0.1)
    assert robot.get_state()["wrench"] == [1, 2, 3, 0, 0, 0]
    robot.inject_fault("force limit")
    fault = robot.get_state()
    assert fault["status"] == "fault"
    assert not fault["enabled"]
    assert fault["wrench"] == [0] * 6
    robot.advance(10)
    assert robot.get_state() == fault
    with pytest.raises(DeviceError) as caught:
        robot.move_joint([1] * 7)
    assert caught.value.code == FAULT
    robot.clear_fault()
    robot.move_joint([1] * 7, duration_ms=100)
    robot.advance(0.1)
    assert robot.get_state()["finished"]


@pytest.mark.parametrize("action", ["joint", "path", "home", "pause", "resume", "clear", "zero", "tool"])
def test_disconnected_robot_rejects_actions(action):
    device = VirtualRobot()
    actions = {
        "joint": lambda: device.move_joint([0] * 7),
        "path": lambda: device.move_path([{"pose": [0] * 6}]),
        "home": device.move_home,
        "pause": device.pause,
        "resume": device.resume,
        "clear": device.clear_fault,
        "zero": device.zero_ft_sensor,
        "tool": lambda: device.set_tool("tool", [0, 0, 10]),
    }
    before = device.get_state()
    with pytest.raises(DeviceError) as caught:
        actions[action]()
    assert caught.value.code == NOT_CONNECTED
    assert device.get_state() == before


def test_path_is_copied_and_state_does_not_repeat_entire_path(robot):
    points = [{"pose": [10, 0, 0, 0, 0, 0]}]
    robot.move_path(points, duration_ms=100)
    points[0]["pose"][0] = 999
    returned = robot.get_path()
    returned[0]["pose"][0] = -999
    robot.advance(0.1)
    assert robot.get_state()["pose"][0] == 10
    assert robot.get_path()[0]["pose"][0] == 10
    assert "last_path" not in robot.get_state()


def test_tool_offset_and_flange_target_use_rotated_translation(robot):
    robot.set_tool("probe", [10, 0, 0])
    assert robot.get_state()["pose"][:3] == pytest.approx([10, 0, 0])
    assert robot.get_state()["flange_pose"][:3] == pytest.approx([0, 0, 0])
    robot.move_path([{"pose": [100, 200, 300, 0, 0, 90], "pose_target": "flange"}], duration_ms=100)
    robot.advance(0.1)
    state = robot.get_state()
    assert state["pose"] == pytest.approx([100, 210, 300, 0, 0, 90])
    assert state["flange_pose"] == pytest.approx([100, 200, 300, 0, 0, 90])


def test_invalid_init_request_does_not_partially_update_home(robot):
    before = robot.get_state()
    with pytest.raises(DeviceError):
        robot.init(home_pose=[1] * 6, home_joints=[0] * 6)
    assert robot.get_state() == before
    robot.move_home(duration_ms=100)
    robot.advance(0.1)
    assert robot.get_state()["pose"] == [0] * 6


def test_warning_light_semantics_and_cleanup():
    light = VirtualWarningLight()
    with pytest.raises(DeviceError) as caught:
        light.set_state("running")
    assert caught.value.code == NOT_CONNECTED
    light.init()
    for state, color, hz in (("running", "green", 0), ("warning", "yellow", 1), ("error", "red", 2)):
        status = light.set_state(state, message=state)
        assert status["colors"] == [color]
        assert status["buzzer_hz"] == hz
        assert status["buzzer"] is (hz > 0)
        assert status["message"] == state
    light.uninit()
    assert light.get_state() == {
        "connected": False, "state": "off", "colors": [],
        "buzzer": False, "buzzer_hz": 0, "message": "",
    }

"""静态设备配置的真实 loopback HTTP 回归，不访问任何实际硬件。"""

import json
import signal
import threading
import time

import cv2
import numpy as np
import pytest

from industrial_sim.adapters.sensium import SensiumStation, SimulatorServer
from industrial_sim.client import IndustrialSimClient
from industrial_sim.core.config import CellConfig, load_cell_config
from industrial_sim.core.errors import FAULT, INVALID_ARGUMENT, NOT_CONNECTED, TIMEOUT, DeviceError
from industrial_sim.scenarios.sensium_cell import run_cycle


@pytest.fixture
def configured(tmp_path):
    # Deliberately use an asymmetric RGB color so swapping red/blue is observable.
    bgr = np.full((8, 12, 3), (7, 19, 231), np.uint8)
    encoded, data = cv2.imencode(".png", bgr)
    assert encoded
    data.tofile(tmp_path / "sample.png")
    points = [{"pose": [1, 2, 3, 4, 5, 6], "reference_frame": "world", "pose_target": "tcp",
               "is_contact": True, "process": {"valid": True, "tcp_vel": 10,
                                                "tcp_acc": 20, "force": [0, 0, -2]}}]
    document = {
        "version": 1,
        "devices": [
            {"function": "CAM_A", "kind": "camera", "seed": 42, "image_files": ["sample.png"], "loop": True,
             "parameters": {"Width": 12, "Height": 8, "PixelFormat": "RGB8", "TriggerMode": "Software"}},
            {"function": "PLC_A", "kind": "io", "ports": 2},
            {"function": "ARM_A", "kind": "duco", "home_pose": [10, 20, 30, 0, 0, 0],
             "home_joints": [1, 2, 3, 4, 5, 6], "preparatory_pose": [40, 50, 60, 0, 0, 0]},
            {"function": "LIGHT_A", "kind": "warning_light"},
        ],
        "cycle": {"io": "PLC_A", "camera": "CAM_A", "robot": "ARM_A", "light": "LIGHT_A",
                  "arrival_input": 9, "complete_output": 8, "running_output": 9, "fault_output": 10,
                  "points": points, "duration_ms": 80, "arrival_timeout_ms": 50,
                  "camera_timeout_ms": 50, "motion_timeout_ms": 2000},
    }
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return load_cell_config(path), path, bgr


@pytest.fixture
def configured_service(configured):
    config, path, bgr = configured
    server = SimulatorServer(config=config, port=0)
    assert int(server.url.rsplit(":", 1)[1]) != 18650
    with server:
        yield server, IndustrialSimClient(server.url), config, path, bgr
    assert not server._thread.is_alive()
    assert all(not state["connected"] for state in server.station.snapshot().values())
    assert server.station.devices["CAM_A"]._grab_thread is None


def assert_configured_cleanup(server):
    snapshot = server.station.snapshot()
    assert set(snapshot) == {"CAM_A", "PLC_A", "ARM_A", "LIGHT_A"}
    assert all(not state["connected"] for state in snapshot.values())
    assert snapshot["PLC_A"]["outputs"] == [0] * 16
    assert snapshot["LIGHT_A"]["state"] == "off"
    assert server.station.devices["CAM_A"]._grab_thread is None


def test_station_creates_only_declared_devices_without_connecting(configured):
    config, _, _ = configured
    station = SensiumStation(config=config)
    try:
        assert set(station.devices) == {"CAM_A", "PLC_A", "ARM_A", "LIGHT_A"}
        assert all(not state["connected"] for state in station.snapshot().values())
        assert station.devices["CAM_A"].get_width() == 12
        assert station.devices["CAM_A"].get_height() == 8
        assert station._thread is None
    finally:
        station.close()


def test_configured_http_refuses_undeclared_init_and_kind_rebinding(configured_service):
    server, client, _, _, _ = configured_service
    for function in ("UNKNOWN", "IO_0", "CAMERA_0"):
        with pytest.raises(DeviceError) as caught:
            client.call(function, "Init", kind="io")
        assert caught.value.code in (NOT_CONNECTED, INVALID_ARGUMENT)
    with pytest.raises(DeviceError) as caught:
        client.call("ARM_A", "Init", kind="flexiv")
    assert caught.value.code == INVALID_ARGUMENT
    assert set(server.station.snapshot()) == {"CAM_A", "PLC_A", "ARM_A", "LIGHT_A"}
    assert not client.state("ARM_A")["connected"]


@pytest.mark.parametrize("mode,source,expected", [
    ("Continuous", "Line2", "Line2"),
    ("Hardware", "Line3", "Line3"),
    ("Hardware", None, "Line0"),
    ("Software", "Software", "Software"),
])
def test_trigger_source_metadata_preserves_line_and_does_not_enable_continuous_trigger(configured, mode, source, expected):
    config, _, _ = configured
    document = config.to_dict()
    parameters = document["devices"][0]["parameters"]
    parameters["TriggerMode"] = mode
    if source is not None:
        parameters["TriggerSource"] = source
    station = SensiumStation(config=CellConfig.from_dict(document))
    try:
        state = station.snapshot()["CAM_A"]
        assert not state["connected"]
        assert state["parameters"]["TriggerMode"] == mode
        assert state["parameters"]["TriggerSource"] == expected
    finally:
        station.close()


def test_configured_color_file_replay_matches_rgb_metadata(configured_service):
    _, client, _, _, bgr = configured_service
    client.call("CAM_A", "Init", kind="camera")
    client.call("CAM_A", "StartGrabbing")
    for _ in range(2):
        metadata, image = client.get_frames("CAM_A")[0]
        assert metadata["pixel_format"] == "RGB8"
        assert (metadata["height"], metadata["width"]) == (8, 12)
        assert image.dtype == np.uint8
        np.testing.assert_array_equal(image, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def test_configured_robot_home_and_preparatory_options_are_applied(configured_service):
    _, client, _, _, _ = configured_service
    client.call("ARM_A", "Init", kind="duco")
    for command, expected in (("MoveHome", [10, 20, 30, 0, 0, 0]),
                              ("MovePreparatory", [40, 50, 60, 0, 0, 0])):
        client.call("ARM_A", command, duration_ms=20)
        deadline = time.monotonic() + 1.0
        while True:
            state = client.state("ARM_A")
            if state["finished"]:
                break
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert state["pose"] == pytest.approx(expected)
        assert state["joints"] == pytest.approx([1, 2, 3, 4, 5, 6])


def test_configured_cycle_uses_names_points_feedback_and_rgb_images(configured_service, monkeypatch):
    server, client, config, _, bgr = configured_service
    original_call = client.call
    commands = []
    images = []

    def record(device, command, **args):
        result = original_call(device, command, **args)
        commands.append((device, command, args))
        return result

    monkeypatch.setattr(client, "call", record)
    result = run_cycle(client, config=config, image_sink=lambda image: images.append(image.copy()))
    assert result.success and result.code == 0
    assert result.robot["brand"] == "duco"
    assert result.robot["pose"] == pytest.approx([1, 2, 3, 4, 5, 6])
    assert result.io["outputs"][8:11] == [1, 0, 0]
    assert result.io["outputs"][:3] == [0, 0, 0]
    assert result.light["colors"] == ["green"]
    assert {device for device, _, _ in commands} == {"CAM_A", "PLC_A", "ARM_A", "LIGHT_A"}
    submitted = [args for device, command, args in commands if device == "ARM_A" and command == "MovePath"]
    assert submitted == [{"points": config.cycle.points, "duration_ms": 80}]
    assert client.call("ARM_A", "GetPath")["points"] == config.cycle.points
    assert len(images) == 1
    np.testing.assert_array_equal(images[0], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    assert_configured_cleanup(server)


def test_configured_burst_delivers_every_frame_to_consumer(configured_service):
    server, client, config, _, bgr = configured_service
    document = config.to_dict()
    document["cycle"].update(frame_count=3, camera_timeout_ms=500)
    burst = CellConfig.from_dict(document)
    images = []
    result = run_cycle(client, config=burst, image_sink=lambda image: images.append(image.copy()))
    assert result.success
    assert len(result.frames) == len(images) == 3
    assert [frame["frame_id"] for frame in result.frames] == [1, 2, 3]
    for image in images:
        np.testing.assert_array_equal(image, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    assert_configured_cleanup(server)


def test_cycle_rejects_different_remote_device_configuration_before_init(configured_service, monkeypatch):
    server, client, config, _, _ = configured_service
    document = config.to_dict()
    document["devices"][0]["parameters"]["Width"] = 13
    other = CellConfig.from_dict(document)
    commands = []
    original_call = client.call

    def record(device, command, **args):
        commands.append(command)
        return original_call(device, command, **args)

    monkeypatch.setattr(client, "call", record)
    result = run_cycle(client, config=other)
    assert not result.success and result.code == INVALID_ARGUMENT
    assert "Init" not in commands
    assert_configured_cleanup(server)


@pytest.mark.parametrize("options,code", [
    ({"inject_arrival": False}, TIMEOUT),
    ({"camera_timeout": True}, TIMEOUT),
    ({"robot_fault": True}, FAULT),
])
def test_configured_failures_keep_complete_low_alarm_and_cleanup(configured_service, monkeypatch, options, code):
    server, client, config, _, _ = configured_service
    original_call = client.call
    commands = []

    def record(device, command, **args):
        response = original_call(device, command, **args)
        commands.append((device, command, args))
        return response

    monkeypatch.setattr(client, "call", record)
    started = time.monotonic()
    result = run_cycle(client, config=config, **options)
    elapsed = time.monotonic() - started
    assert not result.success and result.code == code
    if code == TIMEOUT:
        assert elapsed < 0.7, "the configured 50ms timeout was replaced by a default timeout"
    assert result.io["outputs"][8:11] == [0, 0, 1]
    assert result.light["colors"] == ["red"]
    assert result.light["buzzer_hz"] == 2
    assert not any(device == "PLC_A" and command == "Set" and args.get("index") == 8 and args.get("value") == 1
                   for device, command, args in commands)
    assert_configured_cleanup(server)


def test_device_only_config_serves_but_cycle_is_rejected_before_init(configured, monkeypatch):
    config, _, _ = configured
    document = config.to_dict()
    document.pop("cycle", None)
    device_only = CellConfig.from_dict(document)
    with SimulatorServer(config=device_only, port=0) as server:
        client = IndustrialSimClient(server.url)
        assert client.health()["service"] == "IndustrialSim"
        operations = []
        original_call = client.call

        def record(device, command, **args):
            operations.append(command)
            return original_call(device, command, **args)

        monkeypatch.setattr(client, "call", record)
        try:
            result = run_cycle(client, config=device_only)
        except ValueError:
            result = None
        if result is not None:
            assert not result.success
        assert "Init" not in operations
        assert_configured_cleanup(server)


def test_default_unconfigured_server_and_cycle_remain_compatible():
    with SimulatorServer(port=0) as server:
        client = IndustrialSimClient(server.url)
        assert set(server.station.devices) == {"CAMERA_0", "IO_0", "WARNING_LIGHT_0", "FLEXIV_0", "DUCO_0", "KUKA_0"}
        client.call("CUSTOM_DYNAMIC", "Init", kind="io")
        assert client.state("CUSTOM_DYNAMIC")["connected"]
        client.call("CUSTOM_DYNAMIC", "UnInit")
        result = run_cycle(client)
        assert result.success and result.robot["brand"] == "flexiv"
        assert result.io["outputs"][:3] == [1, 0, 0]
        assert all(not state["connected"] for state in server.station.snapshot().values())


@pytest.mark.parametrize("frame_count", [1, 3])
def test_cli_demo_config_exports_original_png_colors(configured, tmp_path, capsys, frame_count):
    from industrial_sim.__main__ import main

    config, path, bgr = configured
    document = config.to_dict()
    document["cycle"].update(frame_count=frame_count, camera_timeout_ms=500)
    path.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "示例导出"
    assert main(["demo", "--config", str(path), "--output", str(output)]) == 0
    capsys.readouterr()
    exported = list(output.glob("*.png"))
    assert len(exported) == frame_count
    for file in exported:
        image = cv2.imdecode(np.fromfile(file, np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(image, bgr)
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["success"]
    assert set(result["artifacts"]["images"]) == {file.name for file in exported}


def test_cli_serve_uses_config_and_random_port_then_shuts_down(configured, monkeypatch, capsys):
    import industrial_sim.__main__ as entry

    config, path, _ = configured
    ready = threading.Event()
    done = threading.Event()
    servers = []
    handlers = {}
    results = []
    errors = []

    def create_server(**args):
        server = SimulatorServer(**args)
        servers.append(server)
        ready.set()
        return server

    def register_handler(number, callback):
        handlers[number] = callback

    def run():
        try:
            results.append(entry.main(["serve", "--config", str(path), "--port", "0"]))
        except Exception as error:
            errors.append(error)
        finally:
            done.set()

    monkeypatch.setattr(entry, "SimulatorServer", create_server)
    monkeypatch.setattr(entry.signal, "signal", register_handler)
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        assert ready.wait(1.0)
        server = servers[0]
        assert int(server.url.rsplit(":", 1)[1]) != 18650
        client = IndustrialSimClient(server.url)
        assert client.health()["service"] == "IndustrialSim"
        assert client.configuration() == {"configured": True, "config": config.to_dict()}
    finally:
        if signal.SIGINT in handlers:
            handlers[signal.SIGINT](signal.SIGINT, None)
    assert done.wait(2.0)
    worker.join()
    assert errors == [] and results == [0]
    assert all(not state["connected"] for state in server.station.snapshot().values())
    assert not server._thread.is_alive()
    capsys.readouterr()

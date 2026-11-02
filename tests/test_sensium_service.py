"""使用随机 loopback 端口验证 HTTP 与场景；不验收厂商 SDK/Socket 协议。"""

import base64
import http.client
import json
import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np
import pytest

from industrial_sim.adapters.sensium import SensiumStation, SimulatorServer, validate_url
from industrial_sim.client import IndustrialSimClient
from industrial_sim.core.errors import (
    BUSY,
    CANCELLED,
    FAILURE,
    FAULT,
    INVALID_ARGUMENT,
    NOT_CONNECTED,
    TIMEOUT,
    UNSUPPORTED,
    DeviceError,
)
from industrial_sim.scenarios.sensium_cell import run_cycle


@pytest.fixture
def service():
    station = SensiumStation()
    station.devices["CAMERA_0"].set_width(12)
    station.devices["CAMERA_0"].set_height(8)
    server = SimulatorServer(port=0, station=station)
    assert urlparse(server.url).port != 18650
    with server:
        yield server, IndustrialSimClient(server.url)
    assert not server._thread.is_alive()
    assert not station._thread.is_alive()
    assert all(not state["connected"] for state in station.snapshot().values())
    assert station.devices["CAMERA_0"]._grab_thread is None


def init_camera(client, pixel_format="Mono8", buffer_num=10):
    client.call("CAMERA_0", "Init", kind="camera")
    client.call("CAMERA_0", "SetParameter", name="TriggerMode", value="Software")
    client.call("CAMERA_0", "SetParameter", name="PixelFormat", value=pixel_format)
    client.call("CAMERA_0", "StartGrabbing", buffer_num=buffer_num)


def assert_cleaned(server):
    assert all(not state["connected"] for state in server.station.snapshot().values())
    camera = server.station.devices["CAMERA_0"]
    assert not camera._is_grabbing
    assert camera._grab_thread is None
    assert server.station.devices["IO_0"].get_state()["outputs"] == [0] * 16
    assert server.station.devices["WARNING_LIGHT_0"].get_state()["state"] == "off"


def record_commands(client, monkeypatch):
    commands = []
    original_call = client.call

    def record(device, command, **args):
        result = original_call(device, command, **args)
        commands.append((device, command, args))
        return result

    monkeypatch.setattr(client, "call", record)
    return commands


def complete_was_set(commands):
    return any(
        device == "IO_0" and command == "Set" and args.get("index") == 0 and args.get("value") == 1
        for device, command, args in commands
    )


def raw_request(server, payload, path="/v1/command"):
    parsed = urlparse(server.url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
    try:
        connection.request("POST", path, payload, {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_http_health_state_and_function_kind_binding(service):
    server, client = service
    assert client.health() == {"service": "IndustrialSim", "protocol_version": 1}
    for command in ("GetState", "Init"):
        with pytest.raises(DeviceError) as caught:
            client.call("CUSTOM_FUNCTION", command)
        assert caught.value.code == NOT_CONNECTED
    assert client.call("CUSTOM_FUNCTION", "Init", kind="io")["connected"]
    with pytest.raises(DeviceError) as caught:
        client.call("CUSTOM_FUNCTION", "Init", kind="duco")
    assert caught.value.code == INVALID_ARGUMENT
    assert "inputs" in client.state("CUSTOM_FUNCTION")
    with pytest.raises(DeviceError) as caught:
        client.call("OTHER_FUNCTION", "Init", kind="made_up_kind")
    assert caught.value.code == INVALID_ARGUMENT
    assert "OTHER_FUNCTION" not in server.station.snapshot()


@pytest.mark.parametrize("device,kind,command", [
    ("IO_0", "io", "PretendSuccess"),
    ("CAMERA_0", "camera", "PretendSuccess"),
    ("FLEXIV_0", "flexiv", "ExecuteVendorScript"),
    ("WARNING_LIGHT_0", "warning_light", "PretendSuccess"),
])
def test_unsupported_commands_fail_explicitly(service, device, kind, command):
    _, client = service
    client.call(device, "Init", kind=kind)
    with pytest.raises(DeviceError) as caught:
        client.call(device, command)
    assert caught.value.code == UNSUPPORTED


@pytest.mark.parametrize("payload", [
    b"{",
    b"[]",
    b'{"device":"IO_0","command":"Set","args":{"index":0,"value":NaN}}',
    b'{"device":"IO_0","command":"Read","args":[]}',
])
def test_invalid_http_requests_have_structured_error_responses(service, payload):
    server, client = service
    status, response = raw_request(server, payload)
    assert status == 200
    assert response["code"] == INVALID_ARGUMENT
    assert response["data"] == {}
    assert response["message"]
    assert client.health()["service"] == "IndustrialSim"


def test_unknown_http_endpoint_returns_404(service):
    server, _ = service
    status, response = raw_request(server, b"{}", path="/unknown")
    assert status == 404
    assert response["code"] == INVALID_ARGUMENT


@pytest.mark.parametrize("pixel_format,shape,dtype", [
    ("Mono8", (8, 12), np.uint8),
    ("Mono16", (8, 12), np.uint16),
    ("RGB8", (8, 12, 3), np.uint8),
    ("BayerRG8", (8, 12), np.uint8),
])
def test_http_png_dimensions_dtype_and_rgb_channel_order(service, pixel_format, shape, dtype):
    _, client = service
    init_camera(client, pixel_format)
    metadata, image = client.get_frames()[0]
    assert metadata["pixel_format"] == pixel_format
    assert (metadata["height"], metadata["width"]) == shape[:2]
    assert image.shape == shape
    assert image.dtype == dtype
    if pixel_format == "RGB8":
        np.testing.assert_array_equal(image[0, -1], [255, 0, 128])
        # The wire PNG uses normal RGB colors; OpenCV decodes its pixels as BGR.
        raw = client.call("CAMERA_0", "GetFrames", auto_trigger=True)["frames"][0]
        encoded = base64.b64decode(raw["png_base64"], validate=True)
        decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(decoded[0, -1], [128, 0, 255])


def test_http_camera_batch_can_exceed_frame_buffer_capacity(service):
    _, client = service
    init_camera(client, buffer_num=1)
    frames = client.get_frames(frame_count=4, timeout_ms=1000)
    assert len(frames) == 4
    assert [metadata["frame_id"] for metadata, _ in frames] == [1, 2, 3, 4]


def test_http_auto_trigger_discards_preexisting_cached_frame(service):
    server, client = service
    init_camera(client)
    published = threading.Event()
    server.station.devices["CAMERA_0"].register_callback(lambda frame: published.set())
    client.call("CAMERA_0", "TriggerCapture")
    assert published.wait(1.0)
    assert client.get_frames()[0][0]["frame_id"] == 2


def test_http_camera_batch_timeout_is_one_total_deadline(service):
    server, client = service
    init_camera(client)

    def slow_frame():
        time.sleep(0.04)
        return np.zeros((8, 12), np.uint8)

    server.station.devices["CAMERA_0"].set_image_generator(slow_frame)
    with pytest.raises(DeviceError) as caught:
        client.get_frames(frame_count=2, timeout_ms=60)
    assert caught.value.code == TIMEOUT


def test_http_camera_get_frames_can_wait_without_trigger_and_reports_timeout(service):
    _, client = service
    init_camera(client)
    with pytest.raises(DeviceError) as caught:
        client.get_frames(timeout_ms=25, auto_trigger=False)
    assert caught.value.code == TIMEOUT
    client.call("CAMERA_0", "TriggerCapture")
    assert client.get_frames(auto_trigger=False)[0][0]["frame_id"] == 1


def test_overlapping_camera_reads_are_busy_while_stop_can_cancel_wait(service, monkeypatch):
    server, client = service
    init_camera(client)
    camera = server.station.devices["CAMERA_0"]
    waiting = threading.Event()
    done = threading.Event()
    errors = []
    original_wait = camera._condition.wait

    def observe_wait(timeout=None):
        if threading.current_thread().name != "camera-CAMERA_0":
            waiting.set()
        return original_wait(timeout)

    def read():
        try:
            client.get_frames(timeout_ms=5000, auto_trigger=False)
        except DeviceError as error:
            errors.append(error)
        finally:
            done.set()

    monkeypatch.setattr(camera._condition, "wait", observe_wait)
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    assert waiting.wait(1.0)
    with pytest.raises(DeviceError) as caught:
        client.get_frames(timeout_ms=25)
    assert caught.value.code == BUSY
    client.call("CAMERA_0", "StopGrabbing")
    assert done.wait(1.0)
    reader.join()
    assert len(errors) == 1 and errors[0].code == FAILURE


def test_http_io_wait_is_cancelled_by_uninit_on_another_connection(service, monkeypatch):
    server, client = service
    client.call("IO_0", "Init", kind="io")
    waiting = threading.Event()
    done = threading.Event()
    errors = []
    io = server.station.devices["IO_0"]
    original_wait = io._condition.wait

    def observe_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    def read():
        try:
            client.call("IO_0", "ReadWait", index=0, timeout_ms=5000)
        except DeviceError as error:
            errors.append(error)
        finally:
            done.set()

    monkeypatch.setattr(io._condition, "wait", observe_wait)
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    assert waiting.wait(1.0)
    client.call("IO_0", "UnInit")
    client.call("IO_0", "Init", kind="io")
    assert done.wait(1.0)
    reader.join()
    assert len(errors) == 1 and errors[0].code == CANCELLED


def test_server_close_wakes_camera_request_and_closes_all_owned_threads(service, monkeypatch):
    server, client = service
    init_camera(client)
    camera = server.station.devices["CAMERA_0"]
    waiting = threading.Event()
    done = threading.Event()
    errors = []
    original_wait = camera._condition.wait

    def observe_wait(timeout=None):
        if threading.current_thread().name != "camera-CAMERA_0":
            waiting.set()
        return original_wait(timeout)

    def read():
        try:
            client.get_frames(timeout_ms=10000, auto_trigger=False)
        except DeviceError as error:
            errors.append(error)
        finally:
            done.set()

    monkeypatch.setattr(camera._condition, "wait", observe_wait)
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    assert waiting.wait(1.0)
    server.close()
    assert done.wait(1.0)
    reader.join()
    assert len(errors) == 1 and errors[0].code == FAILURE
    assert_cleaned(server)
    with pytest.raises(DeviceError) as caught:
        client.health()
    assert caught.value.code == NOT_CONNECTED


def test_server_close_cleans_up_an_init_that_was_already_inflight(service, monkeypatch):
    server, client = service
    camera = server.station.devices["CAMERA_0"]
    connect_entered = threading.Event()
    release_connect = threading.Event()
    disconnected = threading.Event()
    closed = threading.Event()
    request_done = threading.Event()
    original_connect = camera.connect
    original_disconnect = camera.disconnect
    close_errors = []

    def blocked_connect():
        connect_entered.set()
        assert release_connect.wait(2.0)
        return original_connect()

    def observe_disconnect():
        original_disconnect()
        disconnected.set()

    def initialize():
        try:
            client.call("CAMERA_0", "Init", kind="camera")
        except DeviceError:
            # Rejecting an Init that overlaps shutdown is also a valid result.
            pass
        finally:
            request_done.set()

    def close():
        try:
            server.close()
        except Exception as error:
            close_errors.append(error)
        finally:
            closed.set()

    monkeypatch.setattr(camera, "connect", blocked_connect)
    monkeypatch.setattr(camera, "disconnect", observe_disconnect)
    request = threading.Thread(target=initialize, daemon=True)
    request.start()
    assert connect_entered.wait(1.0)
    closer = threading.Thread(target=close, daemon=True)
    closer.start()
    try:
        # An implementation may clean up immediately or first drain Init. Both
        # orders must leave the device disconnected when close returns.
        disconnected.wait(0.05)
    finally:
        release_connect.set()
    assert request_done.wait(1.0)
    assert closed.wait(1.0)
    request.join()
    closer.join()
    try:
        assert close_errors == []
        assert not camera.is_connected(), "late Init reconnected a device after server cleanup"
        assert_cleaned(server)
    finally:
        original_disconnect()


@pytest.mark.parametrize("robot_kind", ["flexiv", "duco", "kuka"])
def test_run_cycle_models_complete_before_setting_feedback(service, monkeypatch, robot_kind):
    server, client = service
    commands = record_commands(client, monkeypatch)
    result = run_cycle(client, robot_kind=robot_kind)
    assert result.success and result.code == 0
    names = [step["step"] for step in result.steps]
    ordered = ["initialized", "arrival", "captured", "motion_accepted", "motion_completed", "feedback"]
    assert [names.index(name) for name in ordered] == sorted(names.index(name) for name in ordered)
    assert result.robot["brand"] == robot_kind
    assert result.robot["status"] == "completed"
    assert result.robot["progress"] == 1
    assert result.robot["pose"] == pytest.approx([420, 10, 500, 180, 0, 180])
    assert result.io["outputs"][:3] == [1, 0, 0]
    assert result.light["colors"] == ["green"]
    assert complete_was_set(commands)
    assert_cleaned(server)


@pytest.mark.parametrize("options,code", [
    ({"inject_arrival": False, "arrival_timeout_ms": 25}, TIMEOUT),
    ({"camera_timeout": True, "camera_timeout_ms": 25}, TIMEOUT),
    ({"robot_fault": True}, FAULT),
    ({"motion_timeout_ms": 1}, TIMEOUT),
])
def test_cycle_faults_never_set_complete_output_and_alarm_red(service, monkeypatch, options, code):
    server, client = service
    commands = record_commands(client, monkeypatch)
    result = run_cycle(client, **options)
    assert not result.success and result.code == code
    assert result.error
    assert not complete_was_set(commands)
    assert result.io["outputs"][:3] == [0, 0, 1]
    assert result.light["colors"] == ["red"]
    assert result.light["buzzer_hz"] == 2
    assert result.robot["status"] != "completed"
    assert_cleaned(server)


def test_frame_consumer_exception_uses_normal_alarm_and_cleanup_path(service, monkeypatch):
    server, client = service
    commands = record_commands(client, monkeypatch)

    def fail(image):
        raise RuntimeError("vision consumer failed")

    result = run_cycle(client, image_sink=fail)
    assert not result.success and result.code == FAILURE
    assert "vision consumer failed" in result.error
    assert not complete_was_set(commands)
    assert result.io["outputs"][:3] == [0, 0, 1]
    assert result.light["colors"] == ["red"]
    assert_cleaned(server)


def test_frame_derived_path_is_submitted_instead_of_sample_trajectory(service):
    server, client = service
    seen = []

    def build(image):
        seen.append(image.copy())
        return [{"pose": [image.shape[1], image.shape[0], 3, 0, 0, 0]}]

    result = run_cycle(client, trajectory_builder=build)
    assert result.success
    assert len(seen) == 1 and seen[0].shape == (8, 12)
    assert result.robot["pose"] == pytest.approx([12, 8, 3, 0, 0, 0])
    assert client.call("FLEXIV_0", "GetPath")["points"] == [{"pose": [12, 8, 3, 0, 0, 0]}]
    assert_cleaned(server)


def test_stop_after_motion_submission_cancels_motion_without_completion_or_error_output(service, monkeypatch):
    server, client = service
    stop_event = threading.Event()
    commands = []
    original_call = client.call

    def stop_after_accept(device, command, **args):
        response = original_call(device, command, **args)
        commands.append((device, command, args))
        if device == "FLEXIV_0" and command == "MovePath":
            stop_event.set()
        return response

    monkeypatch.setattr(client, "call", stop_after_accept)
    result = run_cycle(client, stop_event=stop_event)
    assert not result.success and result.code == CANCELLED
    assert result.robot["status"] == "cancelled"
    assert result.io["outputs"][:3] == [0, 0, 0]
    assert result.light["colors"] == ["yellow"]
    assert result.light["buzzer_hz"] == 1
    assert not complete_was_set(commands)
    assert_cleaned(server)


def test_already_stopped_cycle_initializes_no_devices(service):
    server, client = service
    stop_event = threading.Event()
    stop_event.set()
    result = run_cycle(client, stop_event=stop_event)
    assert not result.success and result.code == CANCELLED
    assert result.io == {} and result.light == {} and result.robot == {}
    assert_cleaned(server)


@pytest.mark.parametrize("url", [
    "http://192.0.2.1:8765",
    "https://127.0.0.1:8765",
    "http://127.0.0.1",
    "http://user:password@127.0.0.1:8765",
    "http://127.0.0.1:8765/other",
])
def test_client_url_is_explicit_loopback_only(url):
    with pytest.raises(ValueError):
        validate_url(url)

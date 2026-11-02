"""相机模拟器的像素、回放与采集生命周期回归。"""

import threading
import time

import cv2
import numpy as np
import pytest

from industrial_sim.devices.hikvision_camera import (
    CameraDeviceManager,
    HikvisionCameraSimulator,
    PixelFormat,
    TriggerMode,
)


@pytest.fixture
def camera():
    instance = HikvisionCameraSimulator("test-camera", seed=17)
    instance.set_width(12)
    instance.set_height(8)
    instance.set_trigger_mode(TriggerMode.SOFTWARE)
    instance.connect()
    yield instance
    instance.disconnect()


def trigger_frame(camera):
    assert camera.soft_trigger()
    return camera.get_frames(timeout=1.0)[0]


def write_image(path, image):
    encoded, data = cv2.imencode(".png", image)
    assert encoded
    data.tofile(path)
    return path


@pytest.mark.parametrize(
    ("pixel_format", "shape", "dtype"),
    [
        (PixelFormat.Mono8, (8, 12), np.uint8),
        (PixelFormat.Mono16, (8, 12), np.uint16),
        (PixelFormat.RGB8, (8, 12, 3), np.uint8),
        (PixelFormat.BayerRG8, (8, 12), np.uint8),
    ],
)
def test_generated_pixels_match_frame_metadata(camera, pixel_format, shape, dtype):
    camera.set_pixel_format(pixel_format)
    assert camera.start_grabbing()
    frame = trigger_frame(camera)
    assert (frame.width, frame.height) == (12, 8)
    assert frame.pixel_format is pixel_format
    assert frame.data.shape == shape
    assert frame.data.dtype == dtype
    assert frame.frame_id == 1
    assert frame.timestamp >= 0


def test_mono8_seed_reproduces_frame_sequence(camera):
    other = HikvisionCameraSimulator("other-camera", seed=17)
    other.set_width(12)
    other.set_height(8)
    other.set_trigger_mode(TriggerMode.SOFTWARE)
    other.connect()
    try:
        assert camera.start_grabbing()
        assert other.start_grabbing()
        first = trigger_frame(camera).data
        np.testing.assert_array_equal(first, trigger_frame(other).data)
        second = trigger_frame(camera).data
        np.testing.assert_array_equal(second, trigger_frame(other).data)
        assert not np.array_equal(first, second)
        assert first[0, 0] <= 10
        assert first[-1, -1] >= 245
    finally:
        other.disconnect()


def test_rgb_gradient_and_bayer_layout_are_consistent(camera):
    camera.set_pixel_format(PixelFormat.RGB8)
    camera.start_grabbing()
    rgb = trigger_frame(camera).data
    np.testing.assert_array_equal(rgb[0, -1], [255, 0, 128])
    np.testing.assert_array_equal(rgb[-1, 0], [0, 255, 128])
    camera.set_pixel_format(PixelFormat.BayerRG8)
    bayer = trigger_frame(camera).data
    np.testing.assert_array_equal(bayer[0::2, 0::2], rgb[0::2, 0::2, 0])
    np.testing.assert_array_equal(bayer[0::2, 1::2], rgb[0::2, 1::2, 1])
    np.testing.assert_array_equal(bayer[1::2, 0::2], rgb[1::2, 0::2, 1])
    np.testing.assert_array_equal(bayer[1::2, 1::2], rgb[1::2, 1::2, 2])


def test_get_frames_does_not_trigger_and_timeout_uses_seconds(camera):
    camera.start_grabbing()
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        camera.get_frames(timeout=0.03)
    assert time.monotonic() - start >= 0.02
    assert trigger_frame(camera).frame_id == 1


@pytest.mark.parametrize("mode", [TriggerMode.SOFTWARE, TriggerMode.HARDWARE])
def test_explicit_trigger_routes_only_to_the_matching_mode(camera, mode):
    camera.set_trigger_mode(mode)
    assert not camera.trigger_capture()
    camera.start_grabbing()
    wrong_trigger = camera.hardware_trigger if mode is TriggerMode.SOFTWARE else camera.soft_trigger
    assert not wrong_trigger()
    assert camera.trigger_capture()
    assert camera.get_frames()[0].frame_id == 1


def test_continuous_capture_and_batch_larger_than_buffer(camera):
    camera.set_trigger_mode(TriggerMode.CONTINUOUS)
    camera.set_frame_rate(120)
    camera.start_grabbing(buffer_num=1)
    for _ in range(20):
        assert camera.trigger_capture()
    assert camera._pending_triggers == 0
    frames = camera.get_frames(frame_count=3)
    assert len(frames) == 3
    assert [frame.frame_id for frame in frames] == sorted(
        {frame.frame_id for frame in frames}
    )


def test_frame_queue_discards_oldest_frames(camera):
    received = threading.Event()
    camera.register_callback(lambda frame: received.set())
    camera.start_grabbing(buffer_num=2)
    for _ in range(5):
        received.clear()
        assert camera.soft_trigger()
        assert received.wait(1.0)
    assert [frame.frame_id for frame in camera.get_frames(2, timeout=0)] == [4, 5]


def test_pending_trigger_queue_is_bounded(camera):
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    calls = []

    def generate():
        entered.set()
        assert release.wait(2.0)
        return np.zeros((8, 12), dtype=np.uint8)

    def on_frame(frame):
        calls.append(frame.frame_id)
        if len(calls) == 3:
            completed.set()

    camera.set_image_generator(generate)
    camera.register_callback(on_frame)
    camera.start_grabbing(buffer_num=2)
    try:
        assert camera.soft_trigger()
        assert entered.wait(1.0)
        assert camera.soft_trigger()
        assert camera.soft_trigger()
        assert not camera.soft_trigger()
    finally:
        release.set()
    assert completed.wait(1.0)
    assert [frame.frame_id for frame in camera.get_frames(2, timeout=0)] == [2, 3]


def test_file_replay_loops_and_resizes_to_configured_dimensions(camera, tmp_path):
    first = write_image(tmp_path / "第一张.png", np.full((2, 3), 20, np.uint8))
    second = write_image(tmp_path / "second.png", np.full((2, 3), 220, np.uint8))
    camera.set_image_files([first, second])
    camera.start_grabbing()
    for value in (20, 220, 20):
        frame = trigger_frame(camera)
        assert frame.data.shape == (camera.get_height(), camera.get_width())
        assert np.all(frame.data == value)


def test_file_replay_without_loop_exhausts_to_timeout(camera, tmp_path):
    path = write_image(tmp_path / "once.png", np.full((8, 12), 73, np.uint8))
    camera.set_image_files([path], loop=False)
    camera.start_grabbing()
    assert np.all(trigger_frame(camera).data == 73)
    assert camera.soft_trigger()
    with pytest.raises(TimeoutError):
        camera.get_frames(timeout=0.03)


def test_replay_converts_bgr_files_to_rgb_and_bayer(camera, tmp_path):
    image = np.full((8, 12, 3), (10, 20, 230), dtype=np.uint8)
    path = write_image(tmp_path / "color.png", image)
    camera.set_image_files([path])
    camera.set_pixel_format(PixelFormat.RGB8)
    camera.start_grabbing()
    rgb = trigger_frame(camera).data
    assert np.all(rgb == (230, 20, 10))
    camera.set_pixel_format(PixelFormat.BayerRG8)
    bayer = trigger_frame(camera).data
    assert np.all(bayer[0::2, 0::2] == 230)
    assert np.all(bayer[0::2, 1::2] == 20)
    assert np.all(bayer[1::2, 0::2] == 20)
    assert np.all(bayer[1::2, 1::2] == 10)


def test_replay_preserves_mono16_pixels(camera, tmp_path):
    image = np.array([[0, 1, 65535], [256, 32768, 60000]], dtype=np.uint16)
    path = write_image(tmp_path / "depth.png", image)
    camera.set_width(3)
    camera.set_height(2)
    camera.set_pixel_format(PixelFormat.Mono16)
    camera.set_image_files([path])
    camera.start_grabbing()
    np.testing.assert_array_equal(trigger_frame(camera).data, image)


def test_invalid_replay_file_preserves_previous_source(camera, tmp_path):
    valid = write_image(tmp_path / "valid.png", np.full((8, 12), 91, np.uint8))
    camera.set_image_files([valid])
    with pytest.raises(FileNotFoundError):
        camera.set_image_files([tmp_path / "missing.png"])
    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not a valid image")
    with pytest.raises(ValueError, match="Cannot decode"):
        camera.set_image_files([invalid])
    invalid.write_bytes(b"")
    with pytest.raises(ValueError, match="Cannot decode"):
        camera.set_image_files([invalid])
    camera.start_grabbing()
    assert np.all(trigger_frame(camera).data == 91)


def test_custom_generator_output_is_normalized_and_copied(camera):
    image = np.full((2, 3, 3), 300, np.int32)
    camera.set_image_generator(lambda: image)
    camera.start_grabbing()
    frame = trigger_frame(camera)
    image[:] = 0
    assert frame.data.shape == (8, 12)
    assert frame.data.dtype == np.uint8
    assert np.all(frame.data == 255)


def test_timeout_injection_discards_buffered_frames_and_recovers(camera):
    published = threading.Event()
    camera.register_callback(lambda frame: published.set())
    camera.start_grabbing()
    camera.soft_trigger()
    assert published.wait(1.0)
    camera.inject_timeout()
    with pytest.raises(TimeoutError):
        camera.get_frames(timeout=0.03)
    assert camera.soft_trigger()
    with pytest.raises(TimeoutError):
        camera.get_frames(timeout=0.03)
    camera.inject_timeout(False)
    assert trigger_frame(camera).frame_id == 2


def test_clear_frames_discards_cache_without_triggering_a_new_frame(camera):
    published = threading.Event()
    camera.register_callback(lambda frame: published.set())
    camera.start_grabbing()
    camera.soft_trigger()
    assert published.wait(1.0)
    camera.clear_frames()
    with pytest.raises(TimeoutError):
        camera.get_frames(timeout=0.03)
    assert trigger_frame(camera).frame_id == 2


def test_clear_frames_discards_inflight_image_and_pending_triggers(camera):
    entered = threading.Event()
    release = threading.Event()
    published = threading.Event()
    generated = []
    callbacks = []

    def generate():
        generated.append(len(generated) + 1)
        entered.set()
        assert release.wait(2.0)
        return np.full((8, 12), generated[-1], np.uint8)

    def on_frame(frame):
        callbacks.append(frame.frame_id)
        published.set()

    camera.set_image_generator(generate)
    camera.register_callback(on_frame)
    camera.start_grabbing()
    try:
        camera.soft_trigger()
        assert entered.wait(1.0)
        assert camera.soft_trigger()
        assert camera.soft_trigger()
        camera.clear_frames()
    finally:
        release.set()
    with pytest.raises(TimeoutError):
        camera.get_frames(timeout=0.03)
    frame = trigger_frame(camera)
    assert frame.frame_id == 2
    assert np.all(frame.data == 2)
    assert published.wait(1.0)
    assert generated == [1, 2]
    assert callbacks == [2]


def test_timeout_injection_discards_an_inflight_frame(camera):
    entered = threading.Event()
    release = threading.Event()

    def generate():
        entered.set()
        assert release.wait(2.0)
        return np.full((8, 12), 42, np.uint8)

    camera.set_image_generator(generate)
    camera.start_grabbing()
    try:
        camera.soft_trigger()
        assert entered.wait(1.0)
        camera.inject_timeout()
    finally:
        release.set()
    with pytest.raises(TimeoutError):
        camera.get_frames(timeout=0.05)
    camera.inject_timeout(False)
    assert np.all(trigger_frame(camera).data == 42)


@pytest.mark.parametrize("operation", ["stop_grabbing", "disconnect"])
def test_stop_and_disconnect_wake_waiting_reader(camera, monkeypatch, operation):
    camera.start_grabbing()
    waiting = threading.Event()
    returned = threading.Event()
    errors = []
    original_wait = camera._condition.wait

    def observe_wait(timeout=None):
        if threading.current_thread().name == "waiting-reader":
            waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(camera._condition, "wait", observe_wait)

    def read():
        try:
            camera.get_frames(timeout=10.0)
        except Exception as error:
            errors.append(error)
        finally:
            returned.set()

    reader = threading.Thread(target=read, name="waiting-reader", daemon=True)
    reader.start()
    assert waiting.wait(1.0)
    getattr(camera, operation)()
    assert returned.wait(1.0)
    reader.join()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)


def test_callback_can_stop_itself_without_later_callbacks(camera):
    stopped = threading.Event()
    calls = []

    def stop_from_callback(frame):
        calls.append("stop")
        camera.stop_grabbing()
        stopped.set()

    camera.register_callback(stop_from_callback)
    camera.register_callback(lambda frame: calls.append("later"))
    camera.start_grabbing()
    camera.soft_trigger()
    assert stopped.wait(1.0)
    camera.stop_grabbing()
    assert calls == ["stop"]
    assert not camera.soft_trigger()


def test_external_stop_waits_for_callback_beyond_one_second(camera):
    entered = threading.Event()
    release = threading.Event()
    stopped = threading.Event()
    calls = []

    def blocking_callback(frame):
        calls.append(frame.frame_id)
        entered.set()
        release.wait(3.0)

    camera.register_callback(blocking_callback)
    camera.start_grabbing()
    camera.soft_trigger()
    assert entered.wait(1.0)

    def stop():
        camera.stop_grabbing()
        stopped.set()

    stopper = threading.Thread(target=stop, daemon=True)
    stopper.start()
    try:
        assert not stopped.wait(1.1)
    finally:
        release.set()
    assert stopped.wait(1.0)
    stopper.join()
    assert calls == [1]
    assert not camera.soft_trigger()
    with pytest.raises(RuntimeError, match="stopped"):
        camera.get_frames(timeout=0)


def test_stop_waits_for_generation_and_discards_late_frame(camera):
    entered = threading.Event()
    release = threading.Event()
    stop_entered = threading.Event()
    stopped = threading.Event()
    callbacks = []

    def generate():
        entered.set()
        assert release.wait(2.0)
        return np.zeros((8, 12), dtype=np.uint8)

    def stop():
        stop_entered.set()
        camera.stop_grabbing()
        stopped.set()

    camera.set_image_generator(generate)
    camera.register_callback(callbacks.append)
    camera.start_grabbing()
    camera.soft_trigger()
    assert entered.wait(1.0)
    stopper = threading.Thread(target=stop, daemon=True)
    stopper.start()
    try:
        assert stop_entered.wait(1.0)
        assert not stopped.wait(0.03)
        assert not camera.start_grabbing()
    finally:
        release.set()
    assert stopped.wait(1.0)
    stopper.join()
    assert callbacks == []
    with pytest.raises(RuntimeError, match="stopped"):
        camera.get_frames(timeout=0)


def test_stopping_low_frame_rate_stream_interrupts_interval_wait(camera):
    camera.set_trigger_mode(TriggerMode.CONTINUOUS)
    camera.set_frame_rate(1)
    camera.start_grabbing()
    camera.get_frames()
    started = time.monotonic()
    camera.stop_grabbing()
    assert time.monotonic() - started < 0.5


def test_format_change_discards_old_inflight_frame(camera):
    entered = threading.Event()
    release = threading.Event()

    def generate():
        entered.set()
        assert release.wait(2.0)
        return np.full((8, 12), 50, np.uint8)

    camera.set_image_generator(generate)
    camera.start_grabbing()
    try:
        camera.soft_trigger()
        assert entered.wait(1.0)
        camera.set_width(5)
        camera.set_height(3)
        camera.set_pixel_format(PixelFormat.RGB8)
    finally:
        release.set()
    with pytest.raises(TimeoutError):
        camera.get_frames(timeout=0.03)
    frame = trigger_frame(camera)
    assert (frame.width, frame.height, frame.pixel_format) == (5, 3, PixelFormat.RGB8)
    assert frame.data.shape == (3, 5, 3)
    assert frame.data.dtype == np.uint8


def test_callbacks_run_without_holding_camera_lock(camera):
    callback_done = threading.Event()
    lock_available = threading.Event()
    results = []

    def callback(frame):
        def read_parameter():
            camera.get_width()
            lock_available.set()

        reader = threading.Thread(target=read_parameter, daemon=True)
        reader.start()
        results.append(lock_available.wait(0.5))
        callback_done.set()

    camera.register_callback(callback)
    camera.start_grabbing()
    camera.soft_trigger()
    assert callback_done.wait(1.0)
    assert results == [True]


def test_callback_failure_and_unregister_do_not_break_acquisition(camera, caplog):
    received = threading.Event()

    def failing_callback(frame):
        raise ValueError("expected callback failure")

    camera.register_callback(failing_callback)
    camera.register_callback(lambda frame: received.set())
    camera.start_grabbing()
    trigger_frame(camera)
    assert received.wait(1.0)
    assert "expected callback failure" in caplog.text
    camera.unregister_callback(failing_callback)
    received.clear()
    assert trigger_frame(camera).frame_id == 2
    assert received.wait(1.0)


def test_generator_failure_is_reported_and_acquisition_can_restart(camera):
    def broken_generator():
        raise ValueError("expected generator failure")

    camera.set_image_generator(broken_generator)
    camera.start_grabbing()
    assert camera.soft_trigger()
    with pytest.raises(RuntimeError, match="acquisition failed") as caught:
        camera.get_frames()
    assert isinstance(caught.value.__cause__, ValueError)
    camera.stop_grabbing()
    camera.set_image_generator(lambda: np.zeros((8, 12), np.uint8))
    assert camera.start_grabbing()
    assert np.all(trigger_frame(camera).data == 0)


def test_repeated_start_stop_has_no_stale_frames_or_callbacks(camera):
    callback_received = threading.Event()
    calls = []

    def callback(frame):
        calls.append(frame.frame_id)
        callback_received.set()

    camera.register_callback(callback)
    for index in range(8):
        callback_received.clear()
        assert camera.start_grabbing()
        assert camera.start_grabbing()
        with pytest.raises(TimeoutError):
            camera.get_frames(timeout=0)
        assert trigger_frame(camera).frame_id == index + 1
        assert callback_received.wait(1.0)
        camera.stop_grabbing()
        camera.stop_grabbing()
        assert len(calls) == index + 1
    assert calls == list(range(1, 9))


@pytest.mark.parametrize("value", [0, -1, 1.5, True, float("nan")])
def test_dimensions_reject_invalid_values(camera, value):
    with pytest.raises(ValueError):
        camera.set_width(value)
    with pytest.raises(ValueError):
        camera.set_height(value)
    assert (camera.get_width(), camera.get_height()) == (12, 8)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_frame_rate_rejects_invalid_values(camera, value):
    with pytest.raises(ValueError):
        camera.set_frame_rate(value)


def test_get_frames_and_buffer_arguments_are_validated(camera):
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            camera.start_grabbing(buffer_num=value)
        with pytest.raises(ValueError):
            camera.get_frames(frame_count=value)
    for value in (-1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            camera.get_frames(timeout=value)


def test_manager_does_not_replace_active_camera():
    manager = CameraDeviceManager()
    camera = manager.add_virtual_camera("manager-idempotent", "first-serial")
    try:
        camera.set_width(12)
        camera.set_height(8)
        camera.set_trigger_mode(TriggerMode.SOFTWARE)
        camera.connect()
        camera.start_grabbing()
        assert manager.add_virtual_camera("manager-idempotent", "other-serial") is camera
        assert camera.serial_number == "first-serial"
        assert trigger_frame(camera).frame_id == 1
    finally:
        manager.remove_virtual_camera("manager-idempotent")
    assert not camera.is_connected()


@pytest.mark.parametrize("operation", ["remove_virtual_camera", "clear_all"])
def test_manager_does_not_hold_lock_while_waiting_for_callback(monkeypatch, operation):
    manager = CameraDeviceManager()
    camera = manager.add_virtual_camera("manager-callback")
    camera.set_width(12)
    camera.set_height(8)
    camera.set_trigger_mode(TriggerMode.SOFTWARE)
    camera.connect()
    entered = threading.Event()
    release = threading.Event()
    disconnect_entered = threading.Event()
    removed = threading.Event()
    original_disconnect = camera.disconnect

    def callback(frame):
        entered.set()
        release.wait(2.0)
        manager.enumerate_cameras()

    def observe_disconnect():
        disconnect_entered.set()
        original_disconnect()

    def remove():
        if operation == "clear_all":
            manager.clear_all()
        else:
            manager.remove_virtual_camera("manager-callback")
        removed.set()

    monkeypatch.setattr(camera, "disconnect", observe_disconnect)
    camera.register_callback(callback)
    camera.start_grabbing()
    camera.soft_trigger()
    assert entered.wait(1.0)
    remover = threading.Thread(target=remove, daemon=True)
    remover.start()
    try:
        assert disconnect_entered.wait(1.0)
    finally:
        release.set()
    assert removed.wait(1.0)
    remover.join()
    assert manager.get_camera("manager-callback") is None
    assert not camera.is_connected()

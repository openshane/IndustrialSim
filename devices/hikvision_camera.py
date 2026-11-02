"""海康 MVS 相机的进程内模拟器，支持连续采集、软件触发和图片回放。"""

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
import logging
import math
from numbers import Integral, Real
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, Iterable, Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class TriggerMode(Enum):
    """触发模式。"""

    CONTINUOUS = "Continuous"
    SOFTWARE = "Software"
    HARDWARE = "Hardware"


class PixelFormat(Enum):
    """RGB8 使用 RGB 通道顺序；BayerRG8 的首行为 RG。"""

    Mono8 = "Mono8"
    Mono16 = "Mono16"
    RGB8 = "RGB8"
    BayerRG8 = "BayerRG8"


@dataclass
class CameraParameters:
    width: int = 2448
    height: int = 2048
    offset_x: int = 0
    offset_y: int = 0
    exposure_time: float = 10000.0
    gain: float = 1.0
    frame_rate: float = 30.0
    pixel_format: PixelFormat = PixelFormat.Mono8
    trigger_mode: TriggerMode = TriggerMode.CONTINUOUS
    trigger_source: str = "Software"


@dataclass
class FrameData:
    width: int
    height: int
    pixel_format: PixelFormat
    timestamp: float
    frame_id: int
    data: np.ndarray


class HikvisionCameraSimulator:
    """模拟 MVS 的连接、参数、触发和帧回调接口。

    采集与回调只在一个工作线程内执行。触发方法返回请求是否受理，
    实际帧通过 ``get_frames`` 或回调取得。缓冲区满时丢弃最旧帧。
    外部停止等待在途采集和回调结束；回调自身停止后不再派发后续回调。
    """

    def __init__(
        self, device_id: str, serial_number: str = None, *, seed: Optional[int] = None
    ):
        self.device_id = device_id
        self.serial_number = serial_number or f"SIM{device_id[-6:]}"
        self.device_name = f"Hikvision Camera {device_id}"
        self._parameters = CameraParameters()
        self._condition = threading.Condition(threading.RLock())
        self._is_connected = False
        self._is_grabbing = False
        self._frame_id = 0
        self._start_time = time.monotonic()
        self._callbacks: list[Callable[[FrameData], None]] = []
        self._grab_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stop_waiters = 0
        self._generation = 0
        self._source_revision = 0
        self._schedule_revision = 0
        self._frames: deque[FrameData] = deque(maxlen=10)
        self._pending_triggers = 0
        self._capture_error: Optional[Exception] = None
        self._timeout_injected = False
        self._rng = np.random.default_rng(seed)
        self._image_generator: Optional[Callable[[], np.ndarray]] = None
        self._replay_images: tuple[np.ndarray, ...] = ()
        self._replay_index = 0
        self._replay_loop = True
        logger.info("Created simulator for %s", self.device_name)

    @staticmethod
    def _positive_integer(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @staticmethod
    def _finite_number(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be a finite number")
        return result

    def _source_changed(self):
        """调用方持有状态锁，丢弃修改前已排队或正在生成的帧。"""
        self._source_revision += 1
        self._frames.clear()
        self._condition.notify_all()

    def _default_image_generator(self) -> np.ndarray:
        with self._condition:
            parameters = replace(self._parameters)
        return self._generate_test_image(parameters)

    def _generate_test_image(self, parameters: CameraParameters) -> np.ndarray:
        width, height = parameters.width, parameters.height
        maximum = 65535 if parameters.pixel_format == PixelFormat.Mono16 else 255
        x = np.linspace(0, maximum, width, dtype=np.float32)[None, :]
        y = np.linspace(0, maximum, height, dtype=np.float32)[:, None]
        if parameters.pixel_format == PixelFormat.Mono16:
            return ((x + y) / 2).astype(np.uint16)
        if parameters.pixel_format == PixelFormat.Mono8:
            noise = self._rng.integers(-10, 10, (height, width), dtype=np.int16)
            return np.clip((x + y) / 2 + noise, 0, 255).astype(np.uint8)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = x.astype(np.uint8)
        rgb[:, :, 1] = y.astype(np.uint8)
        rgb[:, :, 2] = 128
        if parameters.pixel_format == PixelFormat.RGB8:
            return rgb
        return self._to_bayer(rgb)

    @staticmethod
    def _to_bayer(rgb: np.ndarray) -> np.ndarray:
        bayer = np.empty(rgb.shape[:2], dtype=np.uint8)
        bayer[0::2, 0::2] = rgb[0::2, 0::2, 0]
        bayer[0::2, 1::2] = rgb[0::2, 1::2, 1]
        bayer[1::2, 0::2] = rgb[1::2, 0::2, 1]
        bayer[1::2, 1::2] = rgb[1::2, 1::2, 2]
        return bayer

    @staticmethod
    def _convert_image(
        image: np.ndarray, parameters: CameraParameters, *, bgr: bool = False
    ) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Image generator must return a nonempty numpy array")
        if image.ndim != 2 and not (image.ndim == 3 and image.shape[2] in (3, 4)):
            raise ValueError("Image must have one, three or four channels")
        if not np.issubdtype(image.dtype, np.number) or np.issubdtype(
            image.dtype, np.complexfloating
        ):
            raise ValueError("Image must contain real numeric pixels")
        target_dtype = np.uint16 if parameters.pixel_format == PixelFormat.Mono16 else np.uint8
        if image.dtype == np.uint8 and target_dtype == np.uint16:
            image = image.astype(np.uint16) * 257
        elif image.dtype == np.uint16 and target_dtype == np.uint8:
            image = (image // 257).astype(np.uint8)
        elif image.dtype != target_dtype:
            image = np.clip(image, 0, np.iinfo(target_dtype).max).astype(target_dtype)
        if image.ndim == 3:
            if image.shape[2] == 4:
                conversion = cv2.COLOR_BGRA2RGB if bgr else cv2.COLOR_RGBA2RGB
                image = cv2.cvtColor(image, conversion)
            elif bgr:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (parameters.height, parameters.width):
            image = cv2.resize(image, (parameters.width, parameters.height))
        if parameters.pixel_format == PixelFormat.RGB8:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif parameters.pixel_format == PixelFormat.BayerRG8:
            if image.ndim == 3:
                image = HikvisionCameraSimulator._to_bayer(image)
        elif image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.ascontiguousarray(image).copy()

    def set_image_generator(self, generator: Callable[[], np.ndarray]):
        """设置自定义 RGB/灰度生成器，输出转换为当前尺寸和像素格式。"""
        if not callable(generator):
            raise ValueError("Image generator must be callable")
        with self._condition:
            self._image_generator = generator
            self._replay_images = ()
            self._source_changed()

    def set_image_files(self, paths: Iterable[str], loop: bool = True):
        """读取并回放实际图片，按当前相机尺寸和格式输出。

        ``loop=False`` 时图片耗尽后不再出帧，取帧会超时。图片在设置时
        读取，文件损坏或不存在会立即报错，保留原来的图像源。
        """
        if isinstance(paths, (str, bytes, Path)):
            raise ValueError("paths must be a nonempty iterable of image paths")
        images = []
        for path in paths:
            file_path = Path(path)
            if not file_path.is_file():
                raise FileNotFoundError(file_path)
            # imdecode handles Windows paths containing non-ASCII characters.
            try:
                image = cv2.imdecode(
                    np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                )
            except cv2.error as error:
                raise ValueError(f"Cannot decode image: {file_path}") from error
            if image is None:
                raise ValueError(f"Cannot decode image: {file_path}")
            images.append(image)
        if not images:
            raise ValueError("At least one image file is required")
        with self._condition:
            self._replay_images = tuple(images)
            self._replay_index = 0
            self._replay_loop = bool(loop)
            self._image_generator = None
            self._source_changed()

    def inject_timeout(self, enabled: bool = True):
        """模拟丢帧；清空旧缓冲，随后取帧超时，直至取消注入。"""
        with self._condition:
            self._timeout_injected = bool(enabled)
            self._source_changed()

    def clear_frames(self):
        """清除缓存及待处理触发，并使清理前的在途图像失效。"""
        with self._condition:
            self._pending_triggers = 0
            self._source_changed()

    def get_device_info(self) -> Dict[str, str]:
        return {
            "DeviceID": self.device_id,
            "SerialNumber": self.serial_number,
            "DeviceName": self.device_name,
            "Model": "MV-CA016-10GM",
            "Manufacturer": "Hikvision",
            "FirmwareVersion": "V2.0.0",
            "InterfaceType": "GigE",
        }

    def connect(self) -> bool:
        with self._condition:
            self._is_connected = True
            self._condition.notify_all()
        logger.info("%s connected", self.device_name)
        return True

    def disconnect(self):
        self._stop_acquisition(disconnect=True)
        logger.info("%s disconnected", self.device_name)

    def is_connected(self) -> bool:
        with self._condition:
            return self._is_connected

    def start_grabbing(self, buffer_num: int = 10) -> bool:
        buffer_num = self._positive_integer(buffer_num, "buffer_num")
        with self._condition:
            if not self._is_connected or self._stop_waiters:
                return False
            if self._is_grabbing:
                return True
            # A callback may stop its own acquisition, but cannot start a second
            # worker before the original callback and worker have returned.
            if self._grab_thread is not None:
                return False
            self._frames = deque(maxlen=buffer_num)
            self._pending_triggers = 0
            self._capture_error = None
            self._generation += 1
            self._is_grabbing = True
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._grabbing_loop,
                args=(self._generation,),
                name=f"camera-{self.device_id}",
                daemon=True,
            )
            self._grab_thread = thread
            try:
                thread.start()
            except Exception:
                self._grab_thread = None
                self._is_grabbing = False
                self._stop_event.set()
                self._condition.notify_all()
                raise
        logger.info("%s started grabbing", self.device_name)
        return True

    def stop_grabbing(self):
        self._stop_acquisition()
        logger.info("%s stopped grabbing", self.device_name)

    def _stop_acquisition(self, *, disconnect: bool = False):
        with self._condition:
            if disconnect:
                self._is_connected = False
            self._is_grabbing = False
            self._generation += 1
            self._pending_triggers = 0
            self._frames.clear()
            self._stop_event.set()
            self._stop_waiters += 1
            thread = self._grab_thread
            self._condition.notify_all()
        try:
            if thread is not None and thread is not threading.current_thread():
                thread.join()
        finally:
            with self._condition:
                self._stop_waiters -= 1
                self._condition.notify_all()

    def _active(self, generation: int) -> bool:
        return (
            self._is_connected
            and self._is_grabbing
            and self._generation == generation
        )

    def _grabbing_loop(self, generation: int):
        next_frame = time.monotonic()
        schedule_revision = -1
        try:
            while True:
                with self._condition:
                    while True:
                        if not self._active(generation):
                            return
                        if self._schedule_revision != schedule_revision:
                            schedule_revision = self._schedule_revision
                            next_frame = time.monotonic()
                        if self._parameters.trigger_mode == TriggerMode.CONTINUOUS:
                            delay = next_frame - time.monotonic()
                            if delay <= 0:
                                break
                            self._condition.wait(delay)
                        elif self._pending_triggers:
                            self._pending_triggers -= 1
                            break
                        else:
                            self._condition.wait()
                    source_revision = self._source_revision
                    drop_frame = self._timeout_injected
                frame = None if drop_frame else self._capture_frame()
                if frame is not None:
                    with self._condition:
                        publish = (
                            self._active(generation)
                            and not self._timeout_injected
                            and source_revision == self._source_revision
                        )
                        if publish:
                            self._frames.append(frame)
                            self._condition.notify_all()
                    if publish:
                        self._notify_callbacks(frame, generation)
                with self._condition:
                    next_frame = time.monotonic() + 1.0 / self._parameters.frame_rate
        except Exception as error:
            logger.exception("%s acquisition failed", self.device_name)
            with self._condition:
                if self._generation == generation:
                    self._capture_error = error
                    self._is_grabbing = False
                    self._pending_triggers = 0
                    self._frames.clear()
                    self._stop_event.set()
                    self._condition.notify_all()
        finally:
            with self._condition:
                if self._generation == generation and self._is_grabbing:
                    self._is_grabbing = False
                    self._capture_error = RuntimeError("Acquisition worker exited unexpectedly")
                if self._grab_thread is threading.current_thread():
                    self._grab_thread = None
                self._condition.notify_all()

    def _capture_frame(self) -> Optional[FrameData]:
        with self._condition:
            parameters = replace(self._parameters)
            generator = self._image_generator
            replay_image = None
            if self._replay_images:
                if self._replay_index >= len(self._replay_images):
                    if not self._replay_loop:
                        return None
                    self._replay_index = 0
                replay_image = self._replay_images[self._replay_index]
                self._replay_index += 1
        if replay_image is not None:
            image = self._convert_image(replay_image, parameters, bgr=True)
        elif generator is not None:
            image = self._convert_image(generator(), parameters)
        else:
            image = self._generate_test_image(parameters)
        with self._condition:
            self._frame_id += 1
            frame_id = self._frame_id
        return FrameData(
            width=parameters.width,
            height=parameters.height,
            pixel_format=parameters.pixel_format,
            timestamp=time.monotonic() - self._start_time,
            frame_id=frame_id,
            data=image,
        )

    def _queue_trigger(self, mode: TriggerMode) -> bool:
        with self._condition:
            if (
                not self._is_connected
                or not self._is_grabbing
                or self._parameters.trigger_mode != mode
                or self._pending_triggers >= self._frames.maxlen
            ):
                return False
            self._pending_triggers += 1
            self._condition.notify_all()
            return True

    def soft_trigger(self) -> bool:
        """受理软件触发，不等待图像或回调；待处理请求数不超过缓冲帧数。"""
        return self._queue_trigger(TriggerMode.SOFTWARE)

    def hardware_trigger(self) -> bool:
        """模拟到达相机硬件输入端的触发沿。"""
        return self._queue_trigger(TriggerMode.HARDWARE)

    def trigger_capture(self) -> bool:
        """按当前模式显式触发；连续模式沿用正在采集的流，不积压请求。"""
        with self._condition:
            mode = self._parameters.trigger_mode
            if mode == TriggerMode.CONTINUOUS:
                return self._is_connected and self._is_grabbing
            return self._queue_trigger(mode)

    def get_frames(self, frame_count: int = 1, timeout: float = 1.0) -> list[FrameData]:
        """消费已采集的帧，不隐式触发；timeout 单位为秒。

        超时抛 ``TimeoutError``；停止、断开或采集失败抛 ``RuntimeError``。
        多帧请求可以逐帧消费有界缓冲，超时时不返回部分结果。
        """
        frame_count = self._positive_integer(frame_count, "frame_count")
        timeout = self._finite_number(timeout, "timeout")
        if timeout < 0:
            raise ValueError("timeout must be nonnegative")
        deadline = time.monotonic() + timeout
        frames: list[FrameData] = []
        with self._condition:
            generation = self._generation
            source_revision = self._source_revision
            while True:
                if not self._is_connected:
                    raise RuntimeError("Camera is disconnected")
                if self._capture_error is not None:
                    raise RuntimeError("Camera acquisition failed") from self._capture_error
                if not self._is_grabbing or generation != self._generation:
                    raise RuntimeError("Camera acquisition stopped")
                if source_revision != self._source_revision:
                    frames.clear()
                    source_revision = self._source_revision
                while self._frames and len(frames) < frame_count:
                    frames.append(self._frames.popleft())
                if len(frames) == frame_count:
                    return frames
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for {frame_count} frame(s); received {len(frames)}"
                    )
                self._condition.wait(remaining)

    def register_callback(self, callback: Callable[[FrameData], None]):
        if not callable(callback):
            raise ValueError("Callback must be callable")
        with self._condition:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[FrameData], None]):
        with self._condition:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def _notify_callbacks(self, frame: FrameData, generation: int):
        with self._condition:
            callbacks = tuple(self._callbacks)
        for callback in callbacks:
            with self._condition:
                if not self._active(generation):
                    break
                if callback not in self._callbacks:
                    continue
            try:
                callback(frame)
            except Exception:
                logger.exception("%s frame callback failed", self.device_name)

    def get_width(self) -> int:
        with self._condition:
            return self._parameters.width

    def set_width(self, width: int):
        width = self._positive_integer(width, "width")
        with self._condition:
            self._parameters.width = width
            self._source_changed()

    def get_height(self) -> int:
        with self._condition:
            return self._parameters.height

    def set_height(self, height: int):
        height = self._positive_integer(height, "height")
        with self._condition:
            self._parameters.height = height
            self._source_changed()

    def get_exposure_time(self) -> float:
        with self._condition:
            return self._parameters.exposure_time

    def set_exposure_time(self, time_us: float):
        time_us = self._finite_number(time_us, "exposure_time")
        with self._condition:
            self._parameters.exposure_time = max(20.0, min(1000000.0, time_us))

    def get_gain(self) -> float:
        with self._condition:
            return self._parameters.gain

    def set_gain(self, gain: float):
        gain = self._finite_number(gain, "gain")
        with self._condition:
            self._parameters.gain = max(0.0, min(20.0, gain))

    def get_frame_rate(self) -> float:
        with self._condition:
            return self._parameters.frame_rate

    def set_frame_rate(self, fps: float):
        fps = self._finite_number(fps, "frame_rate")
        if fps <= 0:
            raise ValueError("frame_rate must be positive")
        with self._condition:
            self._parameters.frame_rate = max(1.0, min(120.0, fps))
            self._schedule_revision += 1
            self._condition.notify_all()

    def get_trigger_mode(self) -> TriggerMode:
        with self._condition:
            return self._parameters.trigger_mode

    def set_trigger_mode(self, mode: TriggerMode):
        mode = TriggerMode(mode)
        with self._condition:
            self._parameters.trigger_mode = mode
            if mode == TriggerMode.SOFTWARE:
                self._parameters.trigger_source = "Software"
            elif mode == TriggerMode.HARDWARE and self._parameters.trigger_source == "Software":
                self._parameters.trigger_source = "Line0"
            self._pending_triggers = 0
            self._schedule_revision += 1
            self._source_changed()

    def set_trigger_source(self, source: str):
        if source not in ("Software", "Line0", "Line1", "Line2", "Line3"):
            raise ValueError("unsupported trigger source")
        with self._condition:
            self._parameters.trigger_source = source
            if self._parameters.trigger_mode != TriggerMode.CONTINUOUS:
                self._parameters.trigger_mode = TriggerMode.SOFTWARE if source == "Software" else TriggerMode.HARDWARE
            self._pending_triggers = 0
            self._schedule_revision += 1
            self._source_changed()

    def get_pixel_format(self) -> PixelFormat:
        with self._condition:
            return self._parameters.pixel_format

    def set_pixel_format(self, fmt: PixelFormat):
        fmt = PixelFormat(fmt)
        with self._condition:
            self._parameters.pixel_format = fmt
            self._source_changed()

    def get_all_parameters(self) -> Dict[str, Any]:
        with self._condition:
            parameters = replace(self._parameters)
        return {
            "Width": parameters.width,
            "Height": parameters.height,
            "OffsetX": parameters.offset_x,
            "OffsetY": parameters.offset_y,
            "ExposureTime": parameters.exposure_time,
            "Gain": parameters.gain,
            "FrameRate": parameters.frame_rate,
            "PixelFormat": parameters.pixel_format.value,
            "TriggerMode": parameters.trigger_mode.value,
            "TriggerSource": parameters.trigger_source,
        }

    def __repr__(self):
        with self._condition:
            status = "connected" if self._is_connected else "disconnected"
            grabbing = "grabbing" if self._is_grabbing else "idle"
        return f"<HikvisionCamera {self.device_id} {status} {grabbing}>"


class CameraDeviceManager:
    """相机设备管理器；断开和等待线程始终在管理器锁外执行。"""

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._cameras: Dict[str, HikvisionCameraSimulator] = {}
                instance._lock = threading.RLock()
                cls._instance = instance
        return cls._instance

    def add_virtual_camera(self, device_id: str, serial: str = None) -> HikvisionCameraSimulator:
        """按设备 ID 幂等添加，避免替换仍在采集的实例。"""
        with self._lock:
            camera = self._cameras.get(device_id)
            if camera is None:
                camera = HikvisionCameraSimulator(device_id, serial)
                self._cameras[device_id] = camera
        return camera

    def remove_virtual_camera(self, device_id: str):
        with self._lock:
            camera = self._cameras.pop(device_id, None)
        if camera is not None:
            camera.disconnect()

    def enumerate_cameras(self) -> list:
        with self._lock:
            cameras = list(self._cameras.values())
        return [camera.get_device_info() for camera in cameras]

    def get_camera(self, device_id: str) -> Optional[HikvisionCameraSimulator]:
        with self._lock:
            return self._cameras.get(device_id)

    def clear_all(self):
        with self._lock:
            cameras = list(self._cameras.values())
            self._cameras.clear()
        for camera in cameras:
            camera.disconnect()


camera_manager = CameraDeviceManager()


def demo():
    """演示软件触发和帧读取。"""
    camera = camera_manager.add_virtual_camera("00f0a001", "SN000001")
    camera.connect()
    camera.set_trigger_mode(TriggerMode.SOFTWARE)
    camera.start_grabbing()
    try:
        camera.soft_trigger()
        frame = camera.get_frames()[0]
        print(f"收到帧 #{frame.frame_id}: {frame.width}x{frame.height}, {frame.pixel_format.value}")
    finally:
        camera.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()

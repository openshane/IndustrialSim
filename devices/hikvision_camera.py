"""
海康工业相机设备模块
模拟海康 MVS (Machine Vision SDK) 设备的协议和行为
用于在没有实体相机的情况下进行开发和测试
"""

import numpy as np
import threading
import time
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TriggerMode(Enum):
    """触发模式"""
    CONTINUOUS = "Continuous"      # 连续采集
    SOFTWARE = "Software"          # 软件触发
    HARDWARE = "Hardware"          # 硬件触发


class PixelFormat(Enum):
    """像素格式"""
    Mono8 = "Mono8"
    Mono16 = "Mono16"
    RGB8 = "RGB8"
    BayerRG8 = "BayerRG8"


@dataclass
class CameraParameters:
    """相机参数"""
    width: int = 2448
    height: int = 2048
    offset_x: int = 0
    offset_y: int = 0
    exposure_time: float = 10000.0    # 微秒
    gain: float = 1.0
    frame_rate: float = 30.0
    pixel_format: PixelFormat = PixelFormat.Mono8
    trigger_mode: TriggerMode = TriggerMode.CONTINUOUS
    trigger_source: str = "Software"


@dataclass
class FrameData:
    """帧数据"""
    width: int
    height: int
    pixel_format: PixelFormat
    timestamp: float
    frame_id: int
    data: np.ndarray


class HikvisionCameraSimulator:
    """海康工业相机模拟器
    
    模拟海康 MVS SDK 的主要功能：
    - 设备发现和连接
    - 参数设置（曝光、增益、帧率等）
    - 图像采集（连续/触发模式）
    - 图像回调
    """
    
    def __init__(self, device_id: str, serial_number: str = None):
        self.device_id = device_id
        self.serial_number = serial_number or f"SIM{device_id[-6:]}"
        self.device_name = f"Hikvision Camera {device_id}"
        
        self._parameters = CameraParameters()
        self._is_connected = False
        self._is_grabbing = False
        self._frame_id = 0
        self._start_time = time.time()
        
        self._callbacks: list = []
        self._grab_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 模拟图像生成器
        self._image_generator: Callable[[], np.ndarray] = self._default_image_generator
        
        logger.info(f"Created simulator for {self.device_name}")
    
    def _default_image_generator(self) -> np.ndarray:
        """默认图像生成器 - 生成渐变测试图案"""
        w, h = self._parameters.width, self._parameters.height
        fmt = self._parameters.pixel_format
        
        if fmt == PixelFormat.Mono8:
            # 生成渐变 + 噪声
            img = np.zeros((h, w), dtype=np.uint8)
            x = np.linspace(0, 255, w)
            y = np.linspace(0, 255, h)
            xx, yy = np.meshgrid(x, y)
            img = (xx + yy) / 2
            img += np.random.randint(-10, 10, (h, w), dtype=np.uint8)
            return img
            
        elif fmt == PixelFormat.Mono16:
            img = np.zeros((h, w), dtype=np.uint16)
            x = np.linspace(0, 65535, w)
            y = np.linspace(0, 65535, h)
            xx, yy = np.meshgrid(x, y)
            img = (xx + yy) / 2
            return img.astype(np.uint16)
            
        elif fmt == PixelFormat.RGB8:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:, :, 0] = np.linspace(0, 255, w).astype(np.uint8)
            img[:, :, 1] = np.linspace(0, 255, h).astype(np.uint8)
            img[:, :, 2] = 128
            return img
            
        else:  # Bayer
            return self._default_image_generator()
    
    def set_image_generator(self, generator: Callable[[], np.ndarray]):
        """设置自定义图像生成器"""
        self._image_generator = generator
    
    # ==================== SDK 兼容接口 ====================
    
    def get_device_info(self) -> Dict[str, str]:
        """获取设备信息（模拟 MVS SDK 的接口）"""
        return {
            "DeviceID": self.device_id,
            "SerialNumber": self.serial_number,
            "DeviceName": self.device_name,
            "Model": "MV-CA016-10GM",
            "Manufacturer": "Hikvision",
            "FirmwareVersion": "V2.0.0",
            "InterfaceType": "GigE"
        }
    
    def connect(self) -> bool:
        """连接相机"""
        if self._is_connected:
            logger.warning(f"{self.device_name} already connected")
            return True
        
        # 模拟连接延迟
        time.sleep(0.1)
        self._is_connected = True
        logger.info(f"{self.device_name} connected")
        return True
    
    def disconnect(self):
        """断开连接"""
        if self._is_grabbing:
            self.stop_grabbing()
        
        self._is_connected = False
        logger.info(f"{self.device_name} disconnected")
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._is_connected
    
    def start_grabbing(self, buffer_num: int = 10) -> bool:
        """开始采集
        
        Args:
            buffer_num: 缓冲帧数
        """
        if not self._is_connected:
            logger.error("Cannot start grabbing: not connected")
            return False
        
        if self._is_grabbing:
            logger.warning("Already grabbing")
            return True
        
        self._is_grabbing = True
        self._stop_event.clear()
        
        # 启动采集线程
        self._grab_thread = threading.Thread(target=self._grabbing_loop, daemon=True)
        self._grab_thread.start()
        
        logger.info(f"{self.device_name} started grabbing")
        return True
    
    def stop_grabbing(self):
        """停止采集"""
        self._is_grabbing = False
        self._stop_event.set()
        
        if self._grab_thread and self._grab_thread.is_alive():
            self._grab_thread.join(timeout=1.0)
        
        logger.info(f"{self.device_name} stopped grabbing")
    
    def _grabbing_loop(self):
        """采集循环"""
        interval = 1.0 / self._parameters.frame_rate
        
        while not self._stop_event.is_set():
            if self._parameters.trigger_mode == TriggerMode.CONTINUOUS:
                # 连续采集
                frame = self._capture_frame()
                self._notify_callbacks(frame)
                time.sleep(interval)
            else:
                # 触发模式，等待触发
                time.sleep(0.01)
    
    def _capture_frame(self) -> FrameData:
        """捕获一帧"""
        # 生成图像
        image = self._image_generator()
        
        self._frame_id += 1
        timestamp = time.time() - self._start_time
        
        return FrameData(
            width=image.shape[1],
            height=image.shape[0],
            pixel_format=self._parameters.pixel_format,
            timestamp=timestamp,
            frame_id=self._frame_id,
            data=image
        )
    
    def soft_trigger(self) -> bool:
        """软触发一次（模拟软件触发）"""
        if self._parameters.trigger_mode != TriggerMode.SOFTWARE:
            logger.warning("Not in software trigger mode")
            return False
        
        if not self._is_grabbing:
            logger.warning("Not grabbing")
            return False
        
        # 立即触发一帧
        frame = self._capture_frame()
        self._notify_callbacks(frame)
        return True
    
    def register_callback(self, callback: Callable[[FrameData], None]):
        """注册帧回调"""
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[FrameData], None]):
        """注销帧回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _notify_callbacks(self, frame: FrameData):
        """通知所有回调"""
        for cb in self._callbacks:
            try:
                cb(frame)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    # ==================== 参数设置接口 ====================
    
    def get_width(self) -> int:
        return self._parameters.width
    
    def set_width(self, width: int):
        self._parameters.width = width
    
    def get_height(self) -> int:
        return self._parameters.height
    
    def set_height(self, height: int):
        self._parameters.height = height
    
    def get_exposure_time(self) -> float:
        return self._parameters.exposure_time
    
    def set_exposure_time(self, time_us: float):
        """设置曝光时间（微秒）"""
        self._parameters.exposure_time = max(20.0, min(1000000.0, time_us))
    
    def get_gain(self) -> float:
        return self._parameters.gain
    
    def set_gain(self, gain: float):
        """设置增益（0-20 dB）"""
        self._parameters.gain = max(0.0, min(20.0, gain))
    
    def get_frame_rate(self) -> float:
        return self._parameters.frame_rate
    
    def set_frame_rate(self, fps: float):
        """设置帧率"""
        self._parameters.frame_rate = max(1.0, min(120.0, fps))
    
    def get_trigger_mode(self) -> TriggerMode:
        return self._parameters.trigger_mode
    
    def set_trigger_mode(self, mode: TriggerMode):
        """设置触发模式"""
        self._parameters.trigger_mode = mode
    
    def get_pixel_format(self) -> PixelFormat:
        return self._parameters.pixel_format
    
    def set_pixel_format(self, fmt: PixelFormat):
        """设置像素格式"""
        self._parameters.pixel_format = fmt
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """获取所有参数"""
        return {
            "Width": self._parameters.width,
            "Height": self._parameters.height,
            "OffsetX": self._parameters.offset_x,
            "OffsetY": self._parameters.offset_y,
            "ExposureTime": self._parameters.exposure_time,
            "Gain": self._parameters.gain,
            "FrameRate": self._parameters.frame_rate,
            "PixelFormat": self._parameters.pixel_format.value,
            "TriggerMode": self._parameters.trigger_mode.value,
            "TriggerSource": self._parameters.trigger_source
        }
    
    def __repr__(self):
        status = "connected" if self._is_connected else "disconnected"
        grabbing = "grabbing" if self._is_grabbing else "idle"
        return f"<HikvisionCamera {self.device_id} {status} {grabbing}>"


class CameraDeviceManager:
    """相机设备管理器 - 模拟 MVS SDK 的设备发现"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._cameras: Dict[str, HikvisionCameraSimulator] = {}
                    cls._instance._lock = threading.RLock()
        return cls._instance
    
    def add_virtual_camera(self, device_id: str, serial: str = None) -> HikvisionCameraSimulator:
        """添加虚拟相机"""
        with self._lock:
            camera = HikvisionCameraSimulator(device_id, serial)
            self._cameras[device_id] = camera
            logger.info(f"Added virtual camera: {device_id}")
            return camera
    
    def remove_virtual_camera(self, device_id: str):
        """移除虚拟相机"""
        with self._lock:
            if device_id in self._cameras:
                camera = self._cameras[device_id]
                camera.disconnect()
                del self._cameras[device_id]
    
    def enumerate_cameras(self) -> list:
        """枚举可用相机（模拟 SDK 的枚举接口）"""
        return [cam.get_device_info() for cam in self._cameras.values()]
    
    def get_camera(self, device_id: str) -> Optional[HikvisionCameraSimulator]:
        """获取相机实例"""
        return self._cameras.get(device_id)
    
    def clear_all(self):
        """清空所有相机"""
        with self._lock:
            for cam in self._cameras.values():
                cam.disconnect()
            self._cameras.clear()


# 全局管理器
camera_manager = CameraDeviceManager()


# ==================== 使用示例 ====================

def demo():
    """演示如何使用模拟器"""
    logger.info("=" * 50)
    logger.info("Hikvision Camera Simulator Demo")
    logger.info("=" * 50)
    
    # 添加虚拟相机
    cam1 = camera_manager.add_virtual_camera("00f0a001", "SN000001")
    
    # 连接相机
    cam1.connect()
    print(f"设备信息: {cam1.get_device_info()}")
    
    # 设置参数
    cam1.set_exposure_time(20000)  # 20ms
    cam1.set_gain(5.0)
    cam1.set_frame_rate(30)
    cam1.set_pixel_format(PixelFormat.Mono8)
    print(f"参数: {cam1.get_all_parameters()}")
    
    # 注册回调
    frame_count = [0]
    
    def on_frame(frame: FrameData):
        frame_count[0] += 1
        if frame_count[0] % 30 == 0:
            print(f"收到帧 #{frame.frame_id}, {frame.width}x{frame.height}, {frame.pixel_format.value}")
    
    cam1.register_callback(on_frame)
    
    # 开始采集
    cam1.start_grabbing()
    
    # 运行 2 秒
    time.sleep(2)
    
    # 软触发
    cam1.soft_trigger()
    time.sleep(0.5)
    
    # 停止采集
    cam1.stop_grabbing()
    cam1.disconnect()
    
    print(f"共收到 {frame_count[0]} 帧")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
"""
IndustrialSim 核心模块
设备抽象基类和注册机制
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceBase(ABC):
    """设备抽象基类"""
    
    def __init__(self, device_id: str, name: str, **config):
        self.device_id = device_id
        self.name = name
        self.config = config
        self._properties: Dict[str, Any] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._lock = threading.RLock()
        self._last_update = datetime.now()
        
    @property
    def properties(self) -> Dict[str, Any]:
        """设备属性字典"""
        with self._lock:
            return self._properties.copy()
    
    def set_property(self, key: str, value: Any):
        """设置属性"""
        with self._lock:
            old_value = self._properties.get(key)
            self._properties[key] = value
            self._last_update = datetime.now()
            
            # 触发变更回调
            if key in self._callbacks:
                self._callbacks[key](value, old_value)
            
            logger.debug(f"{self.name}: {key} = {value}")
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """获取属性"""
        with self._lock:
            return self._properties.get(key, default)
    
    def register_callback(self, key: str, callback: Callable):
        """注册属性变更回调"""
        self._callbacks[key] = callback
    
    @abstractmethod
    def update(self, delta_time: float):
        """更新设备状态（由外部定时调用）
        
        Args:
            delta_time: 距离上次更新的时间间隔（秒）
        """
        pass
    
    @abstractmethod
    def get_data(self) -> Dict[str, Any]:
        """获取设备数据（用于协议输出）"""
        pass
    
    def __repr__(self):
        return f"<{self.__class__.__name__} '{self.name}' id={self.device_id}>"


class Sensor(DeviceBase):
    """传感器设备 - 只读数据源"""
    
    def __init__(self, device_id: str, name: str, unit: str = "", **config):
        super().__init__(device_id, name, unit=unit, **config)
        self.unit = unit
        self._value = 0.0
        self._min_value = config.get('min', float('-inf'))
        self._max_value = config.get('max', float('inf'))
        
    @property
    def value(self) -> float:
        return self._value
    
    @value.setter
    def value(self, v: float):
        self._value = max(self._min_value, min(self._max_value, v))
        self.set_property('value', self._value)
    
    def update(self, delta_time: float):
        """传感器默认不自动更新，可由子类重写实现模拟功能"""
        pass
    
    def get_data(self) -> Dict[str, Any]:
        return {
            'value': self._value,
            'unit': self.unit,
            'timestamp': self._last_update.isoformat()
        }


class Actuator(DeviceBase):
    """执行器设备 - 可写入的控制目标"""
    
    def __init__(self, device_id: str, name: str, initial_value: Any = 0, **config):
        super().__init__(device_id, name, initial_value=initial_value, **config)
        self._setpoint = initial_value
        
    @property
    def setpoint(self) -> Any:
        return self._setpoint
    
    @setpoint.setter
    def setpoint(self, value: Any):
        self._setpoint = value
        self.set_property('setpoint', value)
        
    def update(self, delta_time: float):
        """执行器默认无动态，可由子类重写实现PID等控制逻辑"""
        pass
    
    def get_data(self) -> Dict[str, Any]:
        return {
            'setpoint': self._setpoint,
            'timestamp': self._last_update.isoformat()
        }


class PLC(DeviceBase):
    """PLC 设备 - 包含多个寄存器的逻辑控制器"""
    
    def __init__(self, device_id: str, name: str, **config):
        super().__init__(device_id, name, **config)
        self._registers: Dict[int, int] = {}  # address -> value
        self._coils: Dict[int, bool] = {}     # address -> bool
        
    def set_register(self, address: int, value: int):
        """设置保持寄存器"""
        self._registers[address] = value & 0xFFFF  # 16-bit
        self.set_property(f'reg_{address}', value)
    
    def get_register(self, address: int, default: int = 0) -> int:
        return self._registers.get(address, default)
    
    def set_coil(self, address: int, value: bool):
        """设置线圈"""
        self._coils[address] = value
        self.set_property(f'coil_{address}', value)
    
    def get_coil(self, address: int, default: bool = False) -> bool:
        return self._coils.get(address, default)
    
    def update(self, delta_time: float):
        """PLC 逻辑更新，可重写实现自定义控制算法"""
        pass
    
    def get_data(self) -> Dict[str, Any]:
        return {
            'registers': self._registers.copy(),
            'coils': self._coils.copy(),
            'timestamp': self._last_update.isoformat()
        }


class DeviceRegistry:
    """设备注册表 - 管理所有设备实例"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._devices: Dict[str, DeviceBase] = {}
                    cls._instance._device_lock = threading.RLock()
        return cls._instance
    
    def register(self, device: DeviceBase):
        """注册设备"""
        with self._device_lock:
            self._devices[device.device_id] = device
            logger.info(f"Registered device: {device.name} ({device.device_id})")
    
    def unregister(self, device_id: str):
        """注销设备"""
        with self._device_lock:
            if device_id in self._devices:
                del self._devices[device_id]
                logger.info(f"Unregistered device: {device_id}")
    
    def get(self, device_id: str) -> Optional[DeviceBase]:
        """获取设备"""
        return self._devices.get(device_id)
    
    def all(self) -> Dict[str, DeviceBase]:
        """获取所有设备"""
        return self._devices.copy()
    
    def clear(self):
        """清空注册表"""
        with self._device_lock:
            self._devices.clear()


# 全局注册表实例
registry = DeviceRegistry()
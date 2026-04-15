"""IndustrialSim - 工业硬件设备仿真平台"""

from industrial_sim.core.device import (
    DeviceBase,
    Sensor,
    Actuator, 
    PLC,
    DeviceRegistry,
    registry
)

__version__ = "0.1.0"
__all__ = [
    "DeviceBase",
    "Sensor", 
    "Actuator",
    "PLC",
    "DeviceRegistry",
    "registry"
]
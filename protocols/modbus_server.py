"""
Modbus TCP/RTU 协议服务器
基于 pymodbus 实现
"""

import asyncio
import logging
from typing import Dict, Optional
from pymodbus.server import ModbusTcpServer, ModbusSerialServer
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusSlaveContext,
    ModbusServerContext
)
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.logging import Log

from industrial_sim.core.device import registry, Sensor, Actuator, PLC

logger = logging.getLogger(__name__)


class ModbusServer:
    """Modbus 服务器包装器"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 502, slave_id: int = 1):
        self.host = host
        self.port = port
        self.slave_id = slave_id
        self._server = None
        self._running = False
        
    def _build_datastore(self):
        """从注册表构建 Modbus 数据存储"""
        # 初始化 100 个保持寄存器和 100 个线圈
        hr = ModbusSequentialDataBlock(0, [0] * 100)
        ir = ModbusSequentialDataBlock(0, [0] * 100)
        co = ModbusSequentialDataBlock(0, [False] * 100)
        di = ModbusSequentialDataBlock(0, [False] * 100)
        
        # 映射设备到 Modbus 地址空间
        address_map = {}
        
        for device_id, device in registry.all().items():
            if isinstance(device, Sensor):
                # 传感器 -> 保持寄存器 (0x03)
                addr = len([d for d in registry.all().values() if isinstance(d, (Sensor, PLC))])
                address_map[device_id] = ('hr', addr)
                device.set_property('_modbus_addr', addr)
                
            elif isinstance(device, Actuator):
                # 执行器 -> 线圈 (0x01)
                addr = len([d for d in registry.all().values() if isinstance(d, Actuator)])
                address_map[device_id] = ('co', addr)
                device.set_property('_modbus_addr', addr)
                
            elif isinstance(device, PLC):
                # PLC -> 多个寄存器
                for reg_addr, value in device._registers.items():
                    hr.setValues(reg_addr, [value])
                for coil_addr, value in device._coils.items():
                    co.setValues(coil_addr, [value])
        
        store = ModbusSlaveContext(
            hr=hr,  # 保持寄存器
            ir=ir,  # 输入寄存器
            co=co,  # 线圈
            di=di   # 离散输入
        )
        return ModbusServerContext(slave=store, single=True)
    
    async def start(self):
        """启动 Modbus TCP 服务器"""
        context = self._build_datastore()
        
        # 设置设备标识
        identity = ModbusDeviceIdentification()
        identity.VendorName = "IndustrialSim"
        identity.ProductCode = "IS"
        identity.VendorUrl = "https://github.com/openshane/IndustrialSim"
        identity.ModelName = "Modbus Server"
        identity.MajorMinorRevision = "1.0.0"
        
        self._server = ModbusTcpServer(
            context=context,
            identity=identity,
            address=(self.host, self.port)
        )
        
        logger.info(f"Starting Modbus TCP server on {self.host}:{self.port}")
        self._running = True
        await self._server.serve_forever()
    
    def stop(self):
        """停止服务器"""
        if self._server:
            self._server.shutdown()
            self._running = False
            logger.info("Modbus server stopped")


class ModbusSimulator:
    """Modbus 仿真器 - 带设备行为模拟"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 502):
        self.server = ModbusServer(host, port)
        self._task = None
        
    async def start(self):
        """启动仿真器"""
        await self.server.start()
        
    async def simulate(self):
        """运行仿真循环（可重写实现自定义模拟逻辑）"""
        while self.server._running:
            for device_id, device in registry.all().items():
                if hasattr(device, 'update'):
                    device.update(0.1)  # 100ms 周期
            await asyncio.sleep(0.1)
    
    def stop(self):
        """停止仿真器"""
        self.server.stop()


# 便捷函数
def create_modbus_device(device_id: str, name: str, device_type: str = "sensor", **config):
    """创建 Modbus 兼容的设备并注册"""
    if device_type == "sensor":
        device = Sensor(device_id, name, **config)
    elif device_type == "actuator":
        device = Actuator(device_id, name, **config)
    elif device_type == "plc":
        device = PLC(device_id, name, **config)
    else:
        raise ValueError(f"Unknown device type: {device_type}")
    
    registry.register(device)
    return device


async def run_modbus_demo():
    """运行 Modbus 演示"""
    # 创建示例设备
    temp_sensor = create_modbus_device("temp_001", "温度传感器", "sensor", unit="°C", min=-50, max=150)
    temp_sensor.value = 25.5
    
    pressure_sensor = create_modbus_device("press_001", "压力传感器", "sensor", unit="MPa", min=0, max=10)
    pressure_sensor.value = 0.8
    
    valve = create_modbus_device("valve_001", "调节阀", "actuator", initial_value=50)
    valve.setpoint = 75
    
    plc = create_modbus_device("plc_001", "PLC控制器", "plc")
    plc.set_register(0, 100)
    plc.set_register(1, 200)
    plc.set_coil(0, True)
    
    # 启动服务器
    simulator = ModbusSimulator()
    asyncio.create_task(simulator.simulate())
    await simulator.start()


if __name__ == "__main__":
    asyncio.run(run_modbus_demo())
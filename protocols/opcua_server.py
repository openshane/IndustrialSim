"""
OPC-UA 协议服务器
基于 python-opcua 实现
"""

import asyncio
import logging
from typing import Dict, Optional
from opcua import UANodeId, UAVariant, VariantType
from opcua.server import Server, Event
from opcua.common.callback import ChangeCallback

from industrial_sim.core.device import registry, Sensor, Actuator, PLC

logger = logging.getLogger(__name__)


class OPCUAServer:
    """OPC-UA 服务器"""
    
    def __init__(self, endpoint: str = "opc.tcp://localhost:4840", uri: str = "http://industrialsim.github.io"):
        self.endpoint = endpoint
        self.uri = uri
        self._server = Server()
        self._nodes = {}  # device_id -> node_id
        self._running = False
        
    async def start(self):
        """启动 OPC-UA 服务器"""
        await self._server.init()
        self._server.set_endpoint(self.endpoint)
        
        # 设置服务器信息
        self._server.set_server_name("IndustrialSim OPC-UA Server")
        
        # 创建 URI 命名空间
        self.idx = self._server.register_namespace(self.uri)
        
        # 创建根节点
        self.objects = self._server.nodes.objects
        
        logger.info(f"Starting OPC-UA server on {self.endpoint}")
        self._running = True
        self._server.start()
        
    def _device_to_opcua(self, device_id: str):
        """将设备映射到 OPC-UA 节点"""
        device = registry.get(device_id)
        if not device:
            return None
        
        # 在 Objects 下创建设备节点
        device_node = self._server.nodes.objects.add_object(self.idx, device.name)
        self._nodes[device_id] = device_node
        
        if isinstance(device, Sensor):
            # 添加值节点
            value_node = device_node.add_variable(self.idx, "Value", device.value)
            value_node.set_writable()
            device.set_property('_opcua_value_node', value_node)
            
            if device.unit:
                unit_node = device_node.add_variable(self.idx, "Unit", device.unit)
                unit_node.set_writable()
                
        elif isinstance(device, Actuator):
            # 添加设定点节点
            setpoint_node = device_node.add_variable(self.idx, "Setpoint", device.setpoint)
            setpoint_node.set_writable()
            device.set_property('_opcua_setpoint_node', setpoint_node)
            
        elif isinstance(device, PLC):
            # 添加寄存器节点
            for addr, value in device._registers.items():
                reg_node = device_node.add_variable(self.idx, f"Register_{addr}", value)
                reg_node.set_writable()
                
            for addr, value in device._coils.items():
                coil_node = device_node.add_variable(self.idx, f"Coil_{addr}", value)
                coil_node.set_writable()
        
        logger.info(f"Mapped device {device.name} to OPC-UA")
        return device_node
    
    def sync_devices(self):
        """同步设备状态到 OPC-UA"""
        for device_id, device in registry.all().items():
            if device_id not in self._nodes:
                self._device_to_opcua(device_id)
            else:
                # 更新值
                if isinstance(device, Sensor):
                    value_node = device.get_property('_opcua_value_node')
                    if value_node:
                        value_node.set_value(device.value)
                        
                elif isinstance(device, Actuator):
                    setpoint_node = device.get_property('_opcua_setpoint_node')
                    if setpoint_node:
                        setpoint_node.set_value(device.setpoint)
    
    def stop(self):
        """停止服务器"""
        if self._server:
            self._server.stop()
            self._running = False
            logger.info("OPC-UA server stopped")


class OPCUASimulator:
    """OPC-UA 仿真器"""
    
    def __init__(self, endpoint: str = "opc.tcp://localhost:4840"):
        self.server = OPCUAServer(endpoint)
        self._task = None
        
    async def start(self):
        """启动仿真器"""
        await self.server.start()
        
    async def simulate(self):
        """运行仿真循环"""
        while self.server._running:
            # 同步设备状态
            self.server.sync_devices()
            
            # 更新设备
            for device in registry.all().values():
                if hasattr(device, 'update'):
                    device.update(0.1)
                    
            await asyncio.sleep(0.1)
    
    def stop(self):
        """停止仿真器"""
        self.server.stop()


def create_opcua_device(device_id: str, name: str, device_type: str = "sensor", **config):
    """创建设备并自动映射到 OPC-UA"""
    from industrial_sim.core.device import Sensor, Actuator, PLC
    
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


async def run_opcua_demo():
    """运行 OPC-UA 演示"""
    # 创建设备
    temp = create_opcua_device("temp_001", "温度传感器", "sensor", unit="°C")
    temp.value = 23.5
    
    humidity = create_opcua_device("hum_001", "湿度传感器", "sensor", unit="%")
    humidity.value = 65.0
    
    motor = create_opcua_device("motor_001", "电机控制器", "actuator", initial_value=0)
    motor.setpoint = 50
    
    # 启动服务器
    simulator = OPCUASimulator()
    await simulator.start()
    
    # 运行仿真
    await simulator.simulate()


if __name__ == "__main__":
    asyncio.run(run_opcua_demo())
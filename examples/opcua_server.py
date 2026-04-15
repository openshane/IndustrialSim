"""
OPC-UA 服务器示例
演示如何使用 IndustrialSim 创建 OPC-UA 兼容的设备
"""

import asyncio
import logging

from industrial_sim.core.device import registry, Sensor, Actuator
from industrial_sim.protocols.opcua_server import OPCUAServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowSensor(Sensor):
    """流量传感器 - 带波动模拟"""
    
    def __init__(self, device_id: str, name: str, **config):
        super().__init__(device_id, name, unit="m³/h", min=0, max=100, **config)
        import random
        self._random = random
        
    def update(self, delta_time: float):
        """模拟流量波动"""
        # 添加小幅随机波动
        noise = self._random.uniform(-0.5, 0.5)
        self._value = max(self._min_value, min(self._max_value, self._value + noise))
        self.set_property('value', self._value)


class ValveActuator(Actuator):
    """阀门执行器 - 带开度控制"""
    
    def __init__(self, device_id: str, name: str, initial_value: float = 0, **config):
        super().__init__(device_id, name, initial_value=initial_value, **config)
        self._current_position = initial_value
        
    def update(self, delta_time: float):
        """模拟阀门动作延迟"""
        # 简单的阀门动作模拟
        diff = self._setpoint - self._current_position
        if abs(diff) > 0.5:
            move = diff * 0.1  # 10% 响应速度
            self._current_position += move
            self.set_property('position', self._current_position)


async def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("IndustrialSim OPC-UA 示例")
    logger.info("=" * 50)
    
    # 创建流量传感器
    flow = FlowSensor("flow_001", "主管道流量计")
    flow.value = 45.0
    registry.register(flow)
    
    # 创建液位传感器
    level = Sensor("level_001", "储罐液位计", unit="m", min=0, max=10)
    level.value = 5.5
    registry.register(level)
    
    # 创建阀门执行器
    valve = ValveActuator("valve_001", "进水阀", initial_value=30)
    valve.setpoint = 70
    registry.register(valve)
    
    # 显示已注册设备
    logger.info("已注册设备:")
    for dev in registry.all().values():
        logger.info(f"  - {dev.name}: {dev.get_data()}")
    
    # 启动 OPC-UA 服务器
    server = OPCUAServer(endpoint="opc.tcp://0.0.0.0:4840")
    await server.start()
    
    logger.info("\nOPC-UA 服务器已启动: opc.tcp://0.0.0.0:4840")
    logger.info("可以使用 OPC-UA 客户端连接测试")
    logger.info("按 Ctrl+C 停止\n")
    
    # 运行仿真循环
    import time
    last_print = time.time()
    while True:
        # 更新所有设备
        for device in registry.all().values():
            device.update(0.1)
        
        # 同步到 OPC-UA
        server.sync_devices()
        
        # 每秒打印状态
        if time.time() - last_print > 2:
            logger.info(f"流量: {flow.value:.2f} m³/h | 液位: {level.value:.2f}m | 阀门: {valve._current_position:.1f}%")
            last_print = time.time()
        
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n停止仿真...")
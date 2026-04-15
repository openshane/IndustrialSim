"""
Modbus 传感器示例
演示如何使用 IndustrialSim 创建 Modbus 兼容的传感器设备
"""

import asyncio
import logging
import random
import time

from industrial_sim.core.device import registry, Sensor, PLC
from industrial_sim.protocols.modbus_server import ModbusServer, ModbusSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemperatureSensor(Sensor):
    """温度传感器 - 带模拟逻辑"""
    
    def __init__(self, device_id: str, name: str, **config):
        super().__init__(device_id, name, unit="°C", min=-20, max=80, **config)
        self._target = 25.0  # 目标温度
        self._rate = 0.5     # 变化速率
        
    def update(self, delta_time: float):
        """模拟温度缓慢变化"""
        # 简单的模拟：向目标温度趋近
        diff = self._target - self._value
        self._value += diff * self._rate * delta_time
        self.set_property('value', self._value)


class PressurePLC(PLC):
    """压力控制器 PLC - 模拟压力控制和报警"""
    
    def __init__(self, device_id: str, name: str, **config):
        super().__init__(device_id, name, **config)
        # 初始化寄存器
        self.set_register(0, 0)    # 压力值
        self.set_register(1, 100)  # 设定点
        self.set_register(2, 0)   # 报警状态
        self.set_coil(0, False)    # 运行状态
        
    def update(self, delta_time: float):
        """简单的 PID 模拟"""
        # 读取压力
        pressure = self.get_register(0)
        setpoint = self.get_register(1)
        
        # 计算偏差
        error = setpoint - pressure
        
        # 简单控制逻辑
        if abs(error) > 10:
            new_pressure = pressure + (1 if error > 0 else -1) * delta_time * 5
            self.set_register(0, int(new_pressure))
        
        # 报警判断
        if abs(error) > 20:
            self.set_register(2, 1)  # 报警
        else:
            self.set_register(2, 0)


async def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("IndustrialSim Modbus 示例")
    logger.info("=" * 50)
    
    # 创建温度传感器
    temp_sensor = TemperatureSensor("temp_001", "车间温度传感器")
    temp_sensor.value = 22.5
    temp_sensor._target = 28.0  # 加热到 28°C
    registry.register(temp_sensor)
    
    # 创建压力 PLC
    plc = PressurePLC("plc_001", "压力控制 PLC")
    plc.set_register(0, 85)  # 当前压力
    plc.set_register(1, 100)  # 目标压力
    registry.register(plc)
    
    # 创建设备状态展示
    logger.info("已注册设备:")
    for dev in registry.all().values():
        logger.info(f"  - {dev.name} ({dev.device_id}): {dev.get_data()}")
    
    # 启动 Modbus 服务器
    server = ModbusServer(host="0.0.0.0", port=502)
    
    async def run_server():
        await server.start()
    
    # 启动服务器任务
    server_task = asyncio.create_task(run_server())
    
    # 仿真循环
    logger.info("\n运行仿真循环... (按 Ctrl+C 退出)")
    logger.info("Modbus TCP 服务器: 0.0.0.0:502")
    
    last_print = time.time()
    while True:
        # 更新所有设备
        for device in registry.all().values():
            device.update(0.1)
        
        # 每秒打印状态
        if time.time() - last_print > 2:
            logger.info(f"温度: {temp_sensor.value:.2f}°C | 压力: {plc.get_register(0)} | 设定点: {plc.get_register(1)}")
            last_print = time.time()
        
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n停止仿真...")
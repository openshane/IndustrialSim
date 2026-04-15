# IndustrialSim - 工业硬件设备仿真平台

[English](README.md) | 中文

## 概述

IndustrialSim 是一个用于模拟工业设备的开源框架，旨在帮助开发者脱离实体硬件进行自动化调试和测试。

## 特性

- 🔌 **多协议支持**: Modbus TCP/RTU, OPC-UA, MQTT, HTTP, TCP
- 🏭 **设备抽象**: 传感器、执行器、PLC、变频器、仪表等
- 🤖 **可编程行为**: 用 Python 脚本定义设备逻辑
- 🔬 **AI 友好**: 专为 CI/CD 和 AI 自动化测试设计
- 📦 **易于扩展**: 插件化的设备驱动架构

## 快速开始

### 安装

```bash
git clone https://github.com/openshane/IndustrialSim.git
cd IndustrialSim
pip install -r requirements.txt
```

### 运行示例

```bash
# 运行 Modbus 传感器示例
python examples/modbus_sensor.py

# 运行 OPC-UA 服务器示例
python examples/opcua_server.py
```

## 架构

```
IndustrialSim/
├── core/              # 核心框架
│   ├── device.py      # 设备基类
│   ├── registry.py    # 设备注册表
│   └── server.py      # 多协议服务器
├── protocols/         # 协议实现
│   ├── modbus/        # Modbus TCP/RTU
│   ├── opcua/         # OPC-UA
│   └── mqtt/          # MQTT
├── devices/           # 设备模型
│   ├── sensor.py     # 传感器
│   ├── actuator.py   # 执行器
│   └── plc.py         # PLC
└── examples/          # 示例代码
```

## 文档

详细文档见 [docs/](docs/) 目录。

## 贡献

欢迎提交 Issue 和 PR！

## License

MIT License
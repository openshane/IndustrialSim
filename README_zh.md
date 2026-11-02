# IndustrialSim - 工业硬件设备仿真平台

[English](README.md) | 中文

## 概述

IndustrialSim 通过代码构建具有状态、时序和故障行为的虚拟工业设备，让开发者脱离实物完成软件联调。首个参考项目是 Sensium/GrindingTool：按它的设备接口语义实现独立的 Python 模型、场景和本地接口，不修改 GrindingTool。

## 特性

- 🏭 **完整联调流程**：虚拟到料 → 相机采图 → 机器人执行 → IO 与信号灯反馈
- 📷 **工业相机**：图片回放、四种像素格式、软硬触发、帧队列、采集超时注入
- 🤖 **机器人行为**：Flexiv/DUCO/KUKA 轴数与状态模型、轨迹进度、暂停恢复、停止及故障
- 💡 **IO 与信号灯**：输入边沿等待、数字输出、绿/黄/红灯及蜂鸣反馈
- 🔌 **项目适配**：本地 HTTP 设备接口；KUKA 另提供现有 Socket/XML 协议对端
- 🧪 **可重复测试**：正常流程、超时、报警、取消与清理；原协议示例作为可选扩展保留
- 🐍 **代码或配置建设备**：Python 数据类与 JSON 单元配置，自定义设备名、点位、图片序列与轨迹

## 快速开始

### 安装

```powershell
git clone https://github.com/openshane/IndustrialSim.git
cd IndustrialSim
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

### 运行示例

```powershell
# 一条命令运行完整的虚拟设备 HTTP 闭环
.\.venv\Scripts\python.exe -m industrial_sim demo --robot flexiv

# 查看 .industrial-sim/last-run/capture.png 与 result.json
# 启动独立设备服务，供脚本或外部客户端使用
.\.venv\Scripts\python.exe -m industrial_sim serve --port 18650
```

### 使用自己的设备定义

```powershell
# 先校验设备、IO点位、相机参数与轨迹，不启动服务
.\.venv\Scripts\python.exe -m industrial_sim validate --config examples/cells/sensium.json

# 从配置运行完整闭环
.\.venv\Scripts\python.exe -m industrial_sim demo --config examples/cells/sensium.json

# 或直接用 Python 数据类构建单元，接入自己的视觉/规划函数
.\.venv\Scripts\python.exe examples/sensium_cell.py
```

参考 [JSON 示例](examples/cells/sensium.json) 与 [Python 示例](examples/sensium_cell.py)。图片路径按 JSON 文件所在目录解析，支持中文目录；`result.json` 记录配置快照和本轮实际图像清单。

当前闭环通过 IndustrialSim 的实际 HTTP 接口完成，尚未接入未修改的 GrindingTool。KUKA 另有现有 Socket/XML 接口的协议对端；Flexiv/DUCO 原厂 SDK 的透明替换尚未实现。完整步骤、接口单位和边界见 [联调指南](docs/sensium-integration.md)。

### 验证与故障演示

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m industrial_sim demo --fault camera-timeout
```

故障演示预期非零退出、完成输出保持为 0，并产生报警反馈。机器人当前模拟接口与执行状态，关节/TCP 分别插值；动力学、碰撞及厂商 SDK 原始协议兼容性不属于此模型的验证范围。

## 文档

详细使用与适配合同见 [Sensium/GrindingTool 联调指南](docs/sensium-integration.md)。

## 贡献

欢迎提交 Issue 和 PR！

## Star History

<a href="https://www.star-history.com/?repos=openshane%2FIndustrialSim&amp;type=date&amp;legend=top-left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=openshane/IndustrialSim&amp;type=Date&amp;theme=dark" />
    <img alt="IndustrialSim Star History" src="https://api.star-history.com/image?repos=openshane/IndustrialSim&amp;type=Date" />
  </picture>
</a>

## License

MIT License

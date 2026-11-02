# Sensium / GrindingTool 虚拟设备联调

IndustrialSim 使用 Python 定义设备状态、时序、图像源和场景，参考 GrindingTool 的现有设备接口语义。当前交付是独立仿真程序和本地 HTTP 合同，**不修改 GrindingTool**。服务只绑定回环地址，不连接实物设备。

## 安装与首条闭环

在 IndustrialSim 目录中，以 PowerShell 执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m industrial_sim demo --robot flexiv --output .industrial-sim/last-run
.\.venv\Scripts\python.exe -m pytest -q
```

`demo` 自行启动随机回环端口，通过实际 HTTP 请求完成：

1. 初始化 IO、相机、机器人和信号灯，清除完成/运行/故障输出
2. 记录 IO 事件序号，注入 DI0 上升沿，按事件等待到料
3. 请求相机采图，通过 PNG 传输并解码图像
4. 提交三点 TCP 轨迹，区分提交成功和执行完成，跟踪 waypoint/progress
5. 仅在对应 motion_id 完成后置 DO0=1、DO1=0，并显示绿灯
6. 保存 `capture.png` 和 `result.json`，反向释放本次初始化的设备

成功时返回码为 0；错误/取消时非 0，DO0 不会被当作完成置位。DO1 是运行中输出，DO2 是故障输出。结果中的设备快照记录释放前反馈，释放后连接关闭、DO 清零、灯关闭。

`--robot duco` 和 `--robot kuka` 使用对应轴数的 Python 行为模型。**KUKA HTTP 模型演示不是现有 KUKA Socket/XML 适配验收**，后者使用下面的 `kuka-peer`。

故障演示（预期返回码为 1）：

```powershell
.\.venv\Scripts\python.exe -m industrial_sim demo --fault camera-timeout --output .industrial-sim/camera-timeout
.\.venv\Scripts\python.exe -m industrial_sim demo --fault robot-fault --output .industrial-sim/robot-fault
.\.venv\Scripts\python.exe -m industrial_sim demo --fault arrival-timeout --output .industrial-sim/arrival-timeout
```

## 用 Python 或 JSON 定义自己的单元

可运行示例：[Python 单元](../examples/sensium_cell.py)、[JSON 单元](../examples/cells/sensium.json)。两种方式使用同一套 `DeviceSpec`、`CycleSpec`、`CellConfig` 校验规则；配置不执行脚本代码，不启动设备线程或打开端口。

```powershell
.\.venv\Scripts\python.exe -m industrial_sim validate --config examples/cells/sensium.json
.\.venv\Scripts\python.exe -m industrial_sim demo --config examples/cells/sensium.json
```

也可以分开启动服务和场景客户端：

```powershell
# 终端一
.\.venv\Scripts\python.exe -m industrial_sim serve --config examples/cells/sensium.json --port 18650
# 终端二
.\.venv\Scripts\python.exe -m industrial_sim demo --config examples/cells/sensium.json --url http://127.0.0.1:18650
```

客户端会检查服务端的设备定义是否与本地配置一致，不会把新配置默默应用到另一个正在运行的单元。`--robot` 用于内置演示，与 `--config` 互斥；配置方式通过 `cycle.robot` 引用设备，其 `kind` 决定机器人类型。

JSON 顶层为 `version: 1`、`devices` 和可选 `cycle`。省略 `cycle` 可启动一个独立设备集合，但不能运行完整闭环。

| 配置位置 | 内容 |
|---|---|
| `devices[].function` / `kind` | 唯一设备名与 `camera/io/warning_light/flexiv/duco/kuka` 类型 |
| 相机 `parameters` | Width、Height、PixelFormat、ExposureTime（µs）、Gain、FrameRate、TriggerMode、TriggerSource |
| 相机 `seed/image_files/loop` | 测试图随机种子、真实图片序列路径、是否循环回放 |
| IO `ports` | 1–32 个八位端口；DI 与 DO 为独立位数组 |
| 机器人 `home_pose/home_joints/preparatory_pose` | 初始化默认目标；关节数须匹配品牌 |
| `cycle.io/camera/robot/light` | 该流程使用的四个设备名 |
| `cycle.arrival_input/complete_output/running_output/fault_output` | DI 到料位及三个独立 DO 反馈位，均从 0 开始 |
| `cycle.points/duration_ms` | 轨迹与总模拟执行时长，位姿单位沿用 HTTP 合同 |
| `cycle.frame_count` | 单次采集 1–64 帧；统一使用本次采图的总超时预算 |
| `cycle.*_timeout_ms` | arrival、camera、motion 的有界等待时间 |

相机图片路径以 JSON 文件的目录为基准，与执行命令时的工作目录无关；支持 Windows 中文路径。图片会按配置 Width/Height 缩放并转为所选像素格式。`TriggerMode` 可取 `Continuous/Software/Hardware`，`TriggerSource` 可取 `Software/Line0..3`；显式模式与来源相互矛盾时报错，连续模式设置来源不会自动启用触发。

未知字段、重复设备名、错误引用、错误关节数、非有限数值、IO 位越界和不存在/不可解码的图片都会给出字段定位，CLI 返回码为 2，不启动仿真。完整流程运行失败仍返回 1。

`result.json` 包含执行前使用的配置快照、各步骤和反馈，以及 `artifacts.images` 本轮实际写入的图片清单。多帧按 `capture.png`、`capture-0002.png` 等导出；RGB8 图片保存时转换为 OpenCV 所需的 BGR 顺序，Mono16 保留位深。

Python 使用方式：

```python
from industrial_sim.core.config import CellConfig, CycleSpec, DeviceSpec
from industrial_sim.adapters.sensium import SimulatorServer
from industrial_sim.client import IndustrialSimClient
from industrial_sim.scenarios.sensium_cell import run_cycle

config = CellConfig(
    devices=(DeviceSpec("io", "io"), DeviceSpec("camera", "camera"),
             DeviceSpec("arm", "duco"), DeviceSpec("light", "warning_light")),
    cycle=CycleSpec(io="io", camera="camera", robot="arm", light="light",
                    points=[{"pose": [400, 0, 500, 180, 0, 180]}]),
)
with SimulatorServer(config=config) as server:
    result = run_cycle(IndustrialSimClient(server.url), config=config)
    assert result.success, result.error
```

构造配置时复制输入数据；启动服务和场景时再校验并建立自己的快照。配置只声明启动模型，不会自动连接设备。配置化服务拒绝未声明 function 的动态 `Init`，无配置的旧演示接口保持动态设备创建行为。

## 与 GrindingTool 的适配边界

当前正常/故障闭环由 IndustrialSim 自己的客户端经过实际 HTTP 完成，不能据此认定未修改的 GrindingTool 已切换为虚拟设备。本次未交付植入 GrindingTool 的 C++ 后端；该项目的源文件、配置和安装目录保持原状。

启动服务：

```powershell
.\.venv\Scripts\python.exe -m industrial_sim serve --port 18650
```

脚本或独立客户端可连接该地址。现有 GrindingTool 不会自动识别 `SENSIUM_INDUSTRIALSIM_URL`，不需要设置此变量。后续接入须使用它原本支持的外部通信入口，或另行开发项目外适配组件；不向原项目自动复制文件、写配置或部署。

KUKA 现有 Socket/XML 路径提供了无需修改源代码的协议联调入口，见下文。Flexiv/DUCO 的原厂 SDK 透明替换、MVS 可直接枚举的虚拟 GigE 相机、板卡驱动及串口设备级仿真尚未实现；HTTP 行为模型不等于这些底层协议实现。

HTTP 的错误码供独立客户端使用；将来接入原项目时，应由项目外适配组件映射现有错误码，并验证运动完成、回调和生命周期合同。

本服务每个 function 同一时刻由一个应用拥有。独立测试请使用独立服务/随机端口，避免多个应用同时初始化和停止同名设备。

## HTTP 合同 v1

健康检查：`GET /health`；设备状态快照：`GET /v1/state`；配置声明：`GET /v1/config`。

所有设备命令统一 `POST /v1/command`：

```json
{"device":"FLEXIV_0","command":"MovePath","args":{"points":[{"pose":[400,0,500,180,0,180],"reference_frame":"base","pose_target":"tcp"}],"duration_ms":300}}
```

```json
{"code":0,"message":"","data":{"motion_id":1}}
```

`code=0` 成功；`-1` 一般失败、`-2` 参数错误、`-3` 未连接、`-4` 忙、`-5` 超时、`-6` 取消、`-7` 故障、`-8` 未实现。有效设备请求即使失败也返回 JSON 响应。`Init`/`UnInit`/`GetDeviceStatus`/`GetState` 为通用命令。

默认 function 为 `CAMERA_0`、`IO_0`、`WARNING_LIGHT_0`、`FLEXIV_0`、`DUCO_0`、`KUKA_0`。自定义 function 首次 `Init` 需带 `args.kind`，可取 `camera/io/warning_light/flexiv/duco/kuka`；绑定后不能改变类型。

### 机器人

- `MoveJoint`：`joints`、`relative`、可选 `duration_ms`
- `MovePath` / `Polish`：`points`、可选 `duration_ms`
- `MoveHome` / `MovePreparatory`：可选 `pose/joints/duration_ms`；`Init` 可提供默认 `home_pose/preparatory_pose/home_joints`
- `Pause` / `Resume` / `Stop` / `RobotStop`；Stop 可带 `motion_id`，旧编号不能停止新运动
- `ZeroFTSensor`、`SetTool(name,offset)`、`GetToolList`、`InjectFault(message)`、`ClearFault`、`GetPath`

位姿顺序 `[x,y,z,rx,ry,rz]`，平移 mm、角度 deg，旋转顺序 `Rz*Ry*Rx`；关节也是 deg，Flexiv 7 轴、DUCO/KUKA 6 轴。点位保留 `reference_frame/pose_target/is_contact/s/t/ext_axes/process` 元数据；过程速度 m/s、加速度 m/s²、力 N。

当前模型的 base/world 是单位变换；workpiece 坐标需调用方预先转换，外轴轨迹明确拒绝。Flexiv 的反馈标为 base，DUCO/KUKA 为 world。工具只模拟平移偏置与 TCP/法兰基础几何；不进行 FK/IK、碰撞、力控动力学或控制器实时性验证。关节和 TCP 分别插值，不能据此认定两者符合真实机械臂运动学。

提交命令只返回 `motion_id`，`GetState` 返回 `status/progress/waypoint_index/moving/finished/program_state`。只有对应编号 `completed/finished=true/moving=false/program_state=stopped` 才是执行完成。暂停不会完成；Stop、UnInit、故障会终止活动运动。

### 相机

`StartGrabbing/StopGrabbing/TriggerCapture/GetFrames/SetParameter/SetImageFiles/InjectTimeout`。

`GetFrames` 参数是 `frame_count/timeout_ms/auto_trigger`。GrindingTool 的现有采图路径没有先调用 TriggerCapture，因此适配器显式传 `auto_trigger=true`。直接使用 Python 相机对象时，`get_frames` 本身不会触发。

响应 `frames` 包含 `width/height/pixel_format/frame_id/timestamp/png_base64`。PNG 保留 Mono8/Mono16，RGB8 PNG 使用正确颜色。Python `IndustrialSimClient.get_frames` 返回与元数据一致的 RGB 数组；C++ 使用 OpenCV 解码为 BGR `cv::Mat`。

`SetImageFiles` 接受 `paths` 和 `loop`，将本地图像序列按当前尺寸/像素格式回放。`InjectTimeout(enabled=true)` 丢弃新帧并清理旧缓存，用于验证采图超时。停止/断开会取消等待，停止完成后不再发布旧帧/回调。用户自定义图像生成器与回调应能返回；Python 无法安全强杀永久阻塞的用户函数。

### IO 与信号灯

IO：`Set(index,value)` 写 DO；`Read(index)`/`ReadReg(offset)` 读 DI；`InjectInput(index,value)` 注入 DI。`ReadWait(index)`/`ReadRegWait(offset)` 等待**下一次匹配输入变化事件**，包括下降沿，不把当前高电平当作事件。超时为毫秒，0 为无限等待。可携带 `after_sequence` 安全衔接订阅与注入；过期游标报错。UnInit 取消旧代次等待，即使设备很快再次 Init 也不会唤醒旧任务继续工作。

信号灯：`AllOff/SetRunningState/SetWarningState/SetErrorState`。运行绿灯、警告黄灯及 1Hz 蜂鸣、故障红灯及 2Hz 蜂鸣；命令可带 `message`。HTTP 模拟语义反馈，串口逐字节回显协议未在此接口中模拟。

## KUKA 的现有 Socket/XML 路径

GrindingTool 是 TCP 服务端，虚拟 KUKA 连接它的本机端口，并请求 socket_server 配置允许的 Function：

```powershell
.\.venv\Scripts\python.exe -m industrial_sim kuka-peer --port 5000 --function Grinding --fields '{"Part":"demo"}'
```

请将端口与 Function 换成实际项目配置。客户端解析 `<Computer>` XML，支持跨 TCP 分片与轨迹批次；每个非最终批次消费完成后发送精确的 `<Robot><Status>TrackProcessEnd</Status></Robot>`，不附加换行，最终批次不 ACK。响应位姿从线上的 `[x,y,z,rz,ry,rx]` 转成 API 顺序，保留 s/t 和外轴数值。`on_batch` 回调可接入自定义消费逻辑；默认是协议接收，不声称执行了真实机器人运动学。

当前 GrindingTool 的请求接收器只有一次最多 1023 字节的 receive，客户端会限制初始请求大小；TCP 本身不保证一次 send 对应一次 receive，该原有接收限制仍需单独处理。

## 用自己的视觉结果构建轨迹

```python
from industrial_sim.adapters.sensium import SimulatorServer
from industrial_sim.client import IndustrialSimClient
from industrial_sim.scenarios.sensium_cell import run_cycle

def build_path(image):
    # 在这里接入实际视觉/规划代码；返回 mm/deg 的 TCP 点位。
    return [{"pose": [400, 0, 500, 180, 0, 180]},
            {"pose": [410, 0, 490, 180, 0, 180]},
            {"pose": [420, 10, 500, 180, 0, 180]}]

with SimulatorServer() as server:
    result = run_cycle(IndustrialSimClient(server.url), trajectory_builder=build_path)
    assert result.success, result.error
```

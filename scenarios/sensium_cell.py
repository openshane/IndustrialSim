"""The first arrival -> capture -> motion -> feedback integration scenario.

Every operation goes through the public local HTTP contract. The scenario owns
its device lifecycle and must not share function names with a second application
controlling the same station. No GrindingTool files are read or modified.
"""

from dataclasses import asdict, dataclass, field, replace
import threading
import time

from industrial_sim.client import IndustrialSimClient
from industrial_sim.core.config import CellConfig, ConfigurationError, CycleSpec
from industrial_sim.core.errors import CANCELLED, FAILURE, FAULT, INVALID_ARGUMENT, TIMEOUT, DeviceError


@dataclass
class CycleResult:
    success: bool = False
    error: str = ""
    code: int = 0
    steps: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    robot: dict = field(default_factory=dict)
    io: dict = field(default_factory=dict)
    light: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def run_cycle(client: IndustrialSimClient, *, robot_kind=None, inject_arrival=True,
              arrival_timeout_ms=None, camera_timeout_ms=None, motion_timeout_ms=None,
              camera_timeout=False, robot_fault=False, stop_event=None, image_sink=None,
              trajectory_builder=None, config=None):
    if robot_kind is not None and robot_kind not in ("flexiv", "duco", "kuka"):
        raise ValueError("robot_kind must be flexiv, duco or kuka")
    if config is not None:
        if not isinstance(config, CellConfig):
            raise ConfigurationError("config must be a CellConfig")
        config = CellConfig.from_dict(config.to_dict())
        if config.cycle is None:
            raise ConfigurationError("cycle: a closed-loop definition is required for demo")
        selected_kind = config.find_device(config.cycle.robot).kind
        if robot_kind is not None and robot_kind != selected_kind:
            raise ConfigurationError("robot_kind conflicts with cycle.robot in the configuration")
        robot_kind, cycle = selected_kind, config.cycle
    else:
        robot_kind = robot_kind or "flexiv"
        cycle = CycleSpec(robot=f"{robot_kind.upper()}_0")
    if robot_kind not in ("flexiv", "duco", "kuka"):
        raise ValueError("robot_kind must be flexiv, duco or kuka")
    overrides = {key: value for key, value in (("arrival_timeout_ms", arrival_timeout_ms),
                 ("camera_timeout_ms", camera_timeout_ms), ("motion_timeout_ms", motion_timeout_ms)) if value is not None}
    cycle = replace(cycle, **overrides)
    arrival_timeout_ms, camera_timeout_ms, motion_timeout_ms = cycle.arrival_timeout_ms, cycle.camera_timeout_ms, cycle.motion_timeout_ms
    result = CycleResult()
    io, camera, robot, light = cycle.io, cycle.camera, cycle.robot, cycle.light
    owned = []
    motion_id = None
    stop_event = stop_event or threading.Event()

    def step(name, **data):
        result.steps.append({"step": name, **data})

    def cancelled():
        if stop_event.is_set():
            raise DeviceError(CANCELLED, "cycle was stopped")

    def snapshot():
        for function, attribute in ((robot, "robot"), (io, "io"), (light, "light")):
            if function in owned:
                setattr(result, attribute, client.state(function))

    try:
        client.health()
        if config is not None:
            remote = client.configuration()
            if not remote.get("configured") or remote.get("config", {}).get("devices") != config.to_dict()["devices"]:
                raise DeviceError(INVALID_ARGUMENT, "start the service with the same device configuration before running this cycle")
        for function, kind in ((io, "io"), (camera, "camera"),
                               (robot, robot_kind), (light, "warning_light")):
            cancelled()
            client.call(function, "Init", kind=kind)
            owned.append(function)
        for index in (cycle.complete_output, cycle.running_output, cycle.fault_output):
            client.call(io, "Set", index=index, value=0)
        client.call(io, "InjectInput", index=cycle.arrival_input, value=0)
        if config is None:
            client.call(camera, "SetParameter", name="TriggerMode", value="Software")
        client.call(camera, "StartGrabbing")
        client.call(light, "SetRunningState")
        step("initialized", robot=robot)

        # Capture the cursor before injecting arrival. This also works when the
        # server receives ReadWait after the edge, without treating a level as an edge.
        sequence = client.state(io)["sequence"]
        if inject_arrival:
            client.call(io, "InjectInput", index=cycle.arrival_input, value=1)
        deadline = time.monotonic() + arrival_timeout_ms / 1000
        while True:
            cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DeviceError(TIMEOUT, "arrival signal timed out")
            try:
                edge = client.call(io, "ReadWait", index=cycle.arrival_input, after_sequence=sequence,
                                   timeout_ms=max(1, min(100, remaining * 1000)))
            except DeviceError as exc:
                if exc.code == TIMEOUT:
                    continue
                raise
            sequence = edge["sequence"]
            if edge["value"] == 1:
                break
        step("arrival", sequence=sequence)
        client.call(io, "Set", index=cycle.running_output, value=1)
        cancelled()

        client.call(camera, "InjectTimeout", enabled=camera_timeout)
        frames = client.get_frames(device=camera, frame_count=cycle.frame_count, timeout_ms=camera_timeout_ms)
        result.frames = [metadata for metadata, _ in frames]
        if image_sink is not None:
            for _, image in frames:
                try:
                    image_sink(image)
                except Exception as exc:
                    raise DeviceError(FAILURE, f"frame consumer failed: {exc}") from exc
        step("captured", frame_id=frames[0][0]["frame_id"])
        cancelled()

        # A small explicit path exercises ordering and progress. A real vision
        # task can replace this list with the trajectory it computed from frames.
        points = cycle.points
        if trajectory_builder is not None:
            try:
                points = trajectory_builder(frames[0][1])
            except Exception as exc:
                raise DeviceError(FAILURE, f"trajectory builder failed: {exc}") from exc
        motion_id = client.call(robot, "MovePath", points=points, duration_ms=cycle.duration_ms)["motion_id"]
        step("motion_accepted", motion_id=motion_id)
        if robot_fault:
            client.call(robot, "InjectFault", message="scenario robot fault")
        deadline = time.monotonic() + motion_timeout_ms / 1000
        previous_index = -1
        while True:
            cancelled()
            state = client.state(robot)
            if state["motion_id"] != motion_id:
                raise DeviceError(CANCELLED, "motion was replaced by another command")
            if state["status"] == "completed":
                step("motion_completed", motion_id=motion_id, progress=state["progress"])
                break
            if state["status"] in ("fault", "cancelled"):
                raise DeviceError(FAULT if state["status"] == "fault" else CANCELLED,
                                  state.get("fault") or "robot motion was cancelled")
            if time.monotonic() >= deadline:
                raise DeviceError(TIMEOUT, "robot did not complete its motion")
            if state["waypoint_index"] != previous_index:
                previous_index = state["waypoint_index"]
                step("motion_progress", waypoint_index=previous_index, progress=state["progress"])
            stop_event.wait(0.01)

        client.call(io, "Set", index=cycle.running_output, value=0)
        client.call(io, "Set", index=cycle.complete_output, value=1)
        client.call(light, "SetRunningState")
        step("feedback", complete_output=1, output_index=cycle.complete_output, light="green")
        result.success = True
        snapshot()
    except DeviceError as exc:
        result.success = False
        result.error, result.code = str(exc), exc.code
        step("failed", code=exc.code, message=str(exc))
        if motion_id is not None:
            try:
                client.call(robot, "Stop", motion_id=motion_id)
            except DeviceError:
                pass
        if io in owned:
            for index, value in ((cycle.complete_output, 0), (cycle.running_output, 0),
                                 (cycle.fault_output, int(exc.code != CANCELLED))):
                try:
                    client.call(io, "Set", index=index, value=value)
                except DeviceError:
                    pass
        if light in owned:
            try:
                client.call(light, "SetWarningState" if exc.code == CANCELLED else "SetErrorState",
                            message=str(exc))
            except DeviceError:
                pass
        try:
            snapshot()
        except DeviceError:
            pass
    finally:
        for function in reversed(owned):
            try:
                client.call(function, "UnInit")
            except DeviceError as exc:
                result.success = False
                result.error = (result.error + "; " if result.error else "") + f"cleanup {function}: {exc}"
                result.code = exc.code
    return result

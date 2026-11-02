"""Validated, data-only definitions of virtual devices and one test cell.

The same dataclasses are usable directly from Python or loaded from JSON. Paths
in JSON are resolved relative to the configuration file, never the process cwd.
Parsing creates no device threads, sockets, SDK instances or physical devices.
"""

from copy import deepcopy
from dataclasses import dataclass, field, fields
import json
import math
from pathlib import Path

import cv2
import numpy as np

from industrial_sim.core.errors import DeviceError
from industrial_sim.devices.robot import VirtualRobot, finite_vector


class ConfigurationError(ValueError):
    pass


def _fail(location, message):
    raise ConfigurationError(f"{location}: {message}")


def _object(value, location):
    if not isinstance(value, dict):
        _fail(location, "expected an object")
    return value


def _unknown(value, allowed, location):
    extra = set(value) - set(allowed)
    if extra:
        _fail(location, "unknown field(s): " + ", ".join(sorted(map(str, extra))))


def _name(value, location):
    if not isinstance(value, str) or not value.strip() or len(value) > 128 or any(ord(c) < 32 for c in value):
        _fail(location, "expected a nonempty name of at most 128 characters")


def _integer(value, minimum, maximum, location):
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(location, f"expected an integer between {minimum} and {maximum}")


def _number(value, minimum, maximum, location):
    if type(value) not in (int, float) or not minimum <= value <= maximum or not math.isfinite(value):
        _fail(location, f"expected a finite number between {minimum} and {maximum}")


def _default_points():
    return [{"pose": [400, 0, 500, 180, 0, 180]},
            {"pose": [410, 0, 490, 180, 0, 180]},
            {"pose": [420, 10, 500, 180, 0, 180]}]


@dataclass(frozen=True)
class DeviceSpec:
    function: str
    kind: str
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        location = f"devices[{self.function!r}]"
        _name(self.function, location + ".function")
        allowed = {
            "camera": {"parameters", "seed", "image_files", "loop"},
            "io": {"ports"}, "warning_light": set(),
            **{brand: {"home_pose", "preparatory_pose", "home_joints"} for brand in VirtualRobot.PROFILES},
        }
        if not isinstance(self.kind, str) or self.kind not in allowed:
            _fail(location + ".kind", "expected camera, io, warning_light, flexiv, duco or kuka")
        options = deepcopy(_object(self.options, location + ".options"))
        _unknown(options, allowed[self.kind], location)
        if self.kind == "io":
            _integer(options.get("ports", 2), 1, 32, location + ".ports")
        elif self.kind == "camera":
            self._validate_camera(options, location)
        elif self.kind in VirtualRobot.PROFILES:
            for key, count in (("home_pose", 6), ("preparatory_pose", 6),
                               ("home_joints", VirtualRobot.PROFILES[self.kind][0])):
                if key in options:
                    try:
                        options[key] = finite_vector(options[key], count, key)
                    except (DeviceError, OverflowError) as exc:
                        _fail(location + "." + key, str(exc))
        object.__setattr__(self, "options", options)

    @staticmethod
    def _validate_camera(options, location):
        _integer(options.get("seed", 0), 0, 2**63 - 1, location + ".seed")
        if type(options.get("loop", True)) is not bool:
            _fail(location + ".loop", "expected a boolean")
        parameters = _object(options.get("parameters", {}), location + ".parameters")
        ranges = {"Width": (1, 16384), "Height": (1, 16384),
                  "ExposureTime": (20, 1000000), "Gain": (0, 20),
                  "FrameRate": (1, 120), "AcquisitionFrameRate": (1, 120)}
        enums = {"PixelFormat": ("Mono8", "Mono16", "RGB8", "BayerRG8"),
                 "TriggerMode": ("Continuous", "Software", "Hardware"),
                 "TriggerSource": ("Software", "Line0", "Line1", "Line2", "Line3"),
                 "AcquisitionMode": ("Continuous",)}
        _unknown(parameters, ranges.keys() | enums.keys(), location + ".parameters")
        for key, value in parameters.items():
            where = location + ".parameters." + key
            if key in ranges:
                (_integer if key in ("Width", "Height") else _number)(value, *ranges[key], where)
            elif not isinstance(value, str) or value not in enums[key]:
                _fail(where, "expected one of " + ", ".join(enums[key]))
        if "FrameRate" in parameters and "AcquisitionFrameRate" in parameters:
            _fail(location + ".parameters", "specify FrameRate or AcquisitionFrameRate, not both")
        mode, source = parameters.get("TriggerMode"), parameters.get("TriggerSource")
        if (mode == "Software" and source not in (None, "Software")) or (mode == "Hardware" and source == "Software"):
            _fail(location + ".parameters.TriggerSource", "does not match TriggerMode")
        paths = options.get("image_files", [])
        if not isinstance(paths, (list, tuple)):
            _fail(location + ".image_files", "expected a list of image paths")
        normalized = []
        for index, value in enumerate(paths):
            where = f"{location}.image_files[{index}]"
            if not isinstance(value, (str, Path)) or not str(value):
                _fail(where, "expected an image path")
            path = Path(value).resolve()
            if not path.is_file():
                _fail(where, f"file does not exist: {path}")
            try:
                image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            except (OSError, ValueError, cv2.error) as exc:
                _fail(where, f"cannot decode image: {path}: {exc}")
            if image is None:
                _fail(where, f"cannot decode image: {path}")
            normalized.append(str(path))
        if "image_files" in options:
            options["image_files"] = normalized

    def to_dict(self):
        return {"function": self.function, "kind": self.kind, **deepcopy(self.options)}


@dataclass(frozen=True)
class CycleSpec:
    io: str = "IO_0"
    camera: str = "CAMERA_0"
    robot: str = "FLEXIV_0"
    light: str = "WARNING_LIGHT_0"
    arrival_input: int = 0
    complete_output: int = 0
    running_output: int = 1
    fault_output: int = 2
    points: list = field(default_factory=_default_points)
    duration_ms: float = 300
    arrival_timeout_ms: float = 1000
    camera_timeout_ms: float = 1000
    motion_timeout_ms: float = 5000
    frame_count: int = 1

    def __post_init__(self):
        assigned = {}
        for key in ("io", "camera", "robot", "light"):
            _name(getattr(self, key), "cycle." + key)
            function = getattr(self, key)
            if function in assigned:
                _fail("cycle." + key, "device function duplicates cycle." + assigned[function])
            assigned[function] = key
        for key in ("arrival_input", "complete_output", "running_output", "fault_output"):
            _integer(getattr(self, key), 0, 255, "cycle." + key)
        if len({self.complete_output, self.running_output, self.fault_output}) != 3:
            _fail("cycle", "complete_output, running_output and fault_output must be distinct")
        _number(self.duration_ms, 1, 3600000, "cycle.duration_ms")
        for key in ("arrival_timeout_ms", "motion_timeout_ms"):
            _number(getattr(self, key), 1, 86400000, "cycle." + key)
        _number(self.camera_timeout_ms, 1, 60000, "cycle.camera_timeout_ms")
        _integer(self.frame_count, 1, 64, "cycle.frame_count")
        if not isinstance(self.points, (list, tuple)) or not 1 <= len(self.points) <= 10000:
            _fail("cycle.points", "expected between 1 and 10000 trajectory points")
        normalized = []
        for index, point in enumerate(self.points):
            where = f"cycle.points[{index}]"
            dof = 7
            if isinstance(point, dict):
                _unknown(point, {"pose", "joints", "reference_frame", "pose_target", "ext_axes", "is_contact", "s", "t", "process"}, where)
                if "joints" in point:
                    joints = point["joints"]
                    if not isinstance(joints, (list, tuple)) or len(joints) not in (6, 7):
                        _fail(where + ".joints", "expected 6 or 7 joints; device kind determines exact count")
                    dof = len(joints)
                if "ext_axes" in point and not isinstance(point["ext_axes"], (list, tuple)):
                    _fail(where + ".ext_axes", "expected a list")
                for key in ("s", "t"):
                    if key in point:
                        _integer(point[key], -(2**31), 2**31 - 1, where + "." + key)
                if "process" in point:
                    process = _object(point["process"], where + ".process")
                    _unknown(process, {"valid", "tcp_vel", "tcp_acc", "force"}, where + ".process")
                    if "valid" in process and type(process["valid"]) is not bool:
                        _fail(where + ".process.valid", "expected a boolean")
                    for key in ("tcp_vel", "tcp_acc"):
                        if key in process:
                            _number(process[key], 0, 1000000, where + ".process." + key)
            try:
                normalized.extend(VirtualRobot("duco" if dof == 6 else "flexiv").validate_path([point]))
            except (DeviceError, OverflowError) as exc:
                _fail(where, str(exc))
        object.__setattr__(self, "points", normalized)

    def to_dict(self):
        return {item.name: deepcopy(getattr(self, item.name)) for item in fields(self)}


@dataclass(frozen=True)
class CellConfig:
    devices: tuple[DeviceSpec, ...]
    cycle: CycleSpec | None = None

    def __post_init__(self):
        if not isinstance(self.devices, (list, tuple)) or not 1 <= len(self.devices) <= 128:
            _fail("devices", "expected between 1 and 128 device definitions")
        if not all(isinstance(device, DeviceSpec) for device in self.devices):
            _fail("devices", "expected DeviceSpec objects; use from_dict for JSON objects")
        devices = tuple(DeviceSpec(device.function, device.kind, device.options) for device in self.devices)
        if len({device.function for device in devices}) != len(devices):
            _fail("devices", "duplicate function name")
        object.__setattr__(self, "devices", devices)
        if self.cycle is None:
            return
        if not isinstance(self.cycle, CycleSpec):
            _fail("cycle", "expected a CycleSpec object")
        cycle = CycleSpec(**self.cycle.to_dict())
        for key, kind in (("io", "io"), ("camera", "camera"), ("light", "warning_light")):
            device = self.find_device(getattr(cycle, key), location="cycle." + key)
            if device.kind != kind:
                _fail("cycle." + key, "device kind must be " + kind)
        robot = self.find_device(cycle.robot, location="cycle.robot")
        if robot.kind not in VirtualRobot.PROFILES:
            _fail("cycle.robot", "device kind must be a robot")
        ports = self.find_device(cycle.io).options.get("ports", 2)
        for key in ("arrival_input", "complete_output", "running_output", "fault_output"):
            _integer(getattr(cycle, key), 0, ports * 8 - 1, "cycle." + key)
        try:
            points = VirtualRobot(robot.kind).validate_path(cycle.points)
        except DeviceError as exc:
            _fail("cycle.points", str(exc))
        object.__setattr__(cycle, "points", points)
        object.__setattr__(self, "cycle", cycle)

    def find_device(self, function, location="cycle"):
        for device in self.devices:
            if device.function == function:
                return device
        _fail(location, f"unknown device function: {function}")

    def to_dict(self):
        document = {"version": 1, "devices": [device.to_dict() for device in self.devices]}
        if self.cycle is not None:
            document["cycle"] = self.cycle.to_dict()
        return document

    @classmethod
    def from_dict(cls, document, base_dir=Path(".")):
        document = deepcopy(_object(document, "$"))
        _unknown(document, {"version", "devices", "cycle"}, "$")
        if type(document.get("version")) is not int or document["version"] != 1:
            _fail("version", "expected supported version 1")
        raw_devices = document.get("devices")
        if not isinstance(raw_devices, list):
            _fail("devices", "expected a list")
        devices = []
        for index, item in enumerate(raw_devices):
            item = _object(item, f"devices[{index}]")
            if "function" not in item or "kind" not in item:
                _fail(f"devices[{index}]", "function and kind are required")
            options = {key: value for key, value in item.items() if key not in ("function", "kind")}
            if "image_files" in options and isinstance(options["image_files"], (list, tuple)):
                options["image_files"] = [str(Path(base_dir) / path) if isinstance(path, (str, Path)) and str(path) else path
                                          for path in options["image_files"]]
            devices.append(DeviceSpec(item["function"], item["kind"], options))
        cycle = None
        if "cycle" in document:
            raw_cycle = _object(document["cycle"], "cycle")
            _unknown(raw_cycle, {item.name for item in fields(CycleSpec)}, "cycle")
            cycle = CycleSpec(**raw_cycle)
        return cls(tuple(devices), cycle)


def load_cell_config(path):
    path = Path(path).resolve()

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("$", "duplicate JSON key: " + key)
            result[key] = value
        return result

    try:
        text = path.read_text(encoding="utf-8-sig")
        document = json.loads(text, object_pairs_hook=unique_keys,
                              parse_constant=lambda value: _fail("$", "nonfinite number: " + value))
        return CellConfig.from_dict(document, base_dir=path.parent)
    except ConfigurationError:
        raise
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc

"""Deterministic robot state/trajectory simulation, without a dynamics solver.

Pose units are mm/deg in [x,y,z,rx,ry,rz] order; joint units are degrees.
The owner advances the model using elapsed seconds. No motion thread or real
robot SDK is created here. Joint and Cartesian interpolation are independent;
this model deliberately does not claim forward/inverse kinematics.
"""

from copy import deepcopy
import math
import threading

from industrial_sim.core.errors import (
    BUSY, CANCELLED, FAULT, INVALID_ARGUMENT, NOT_CONNECTED, UNSUPPORTED, DeviceError,
)


def finite_vector(value, count, name):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise DeviceError(INVALID_ARGUMENT, f"{name} must contain {count} numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise DeviceError(INVALID_ARGUMENT, f"{name} must contain finite numbers")
    return [float(x) for x in value]


class VirtualRobot:
    PROFILES = {"flexiv": (7, "base"), "duco": (6, "world"), "kuka": (6, "world")}

    def __init__(self, brand="flexiv", function=None):
        if brand not in self.PROFILES:
            raise ValueError("unsupported robot brand")
        self.brand = brand
        self.function = function or f"{brand.upper()}_0"
        self.dof, self.pose_frame = self.PROFILES[brand]
        self._lock = threading.RLock()
        self._connected = False
        self._status = "idle"
        self._pose = [0.0] * 6
        self._joints = [0.0] * self.dof
        self._home_pose = [0.0] * 6
        self._preparatory_pose = [0.0] * 6
        self._home_joints = [0.0] * self.dof
        self._motion_id = 0
        self._admission_epoch = 0
        self._progress = 0.0
        self._waypoint_index = -1
        self._points = []
        self._targets = []
        self._start_pose = list(self._pose)
        self._start_joints = list(self._joints)
        self._elapsed = 0.0
        self._duration = 0.0
        self._fault = ""
        self._wrench = [0.0] * 6
        self._tools = {"default": [0.0, 0.0, 0.0]}
        self._tool = "default"

    def init(self, **args):
        # Validate the complete request before mutating a live model.
        updates = {}
        for key, count in (("home_pose", 6), ("preparatory_pose", 6), ("home_joints", self.dof)):
            if key in args:
                updates["_" + key] = finite_vector(args[key], count, key)
        with self._lock:
            if self._status in ("running", "paused"):
                raise DeviceError(BUSY, "cannot reinitialize a moving robot")
            for key, value in updates.items():
                setattr(self, key, value)
            self._connected = True
            self._admission_epoch += 1
            self._status = "fault" if self._fault else "idle"
            return self.get_state()

    def uninit(self):
        with self._lock:
            self.stop()
            self._connected = False

    def _ready(self):
        if not self._connected:
            raise DeviceError(NOT_CONNECTED, "robot is not initialized")
        if self._fault:
            raise DeviceError(FAULT, self._fault)

    def _point(self, item):
        if not isinstance(item, dict):
            item = {"pose": item}
        point = deepcopy(item)
        if point.get("reference_frame", self.pose_frame) not in ("base", "world"):
            raise DeviceError(UNSUPPORTED, "resolve workpiece coordinates before submitting a virtual path")
        if point.get("pose_target", "tcp") not in ("tcp", "flange"):
            raise DeviceError(INVALID_ARGUMENT, "invalid pose_target")
        if point.get("ext_axes"):
            raise DeviceError(UNSUPPORTED, "external axes are not modeled")
        if "pose" not in point and "joints" not in point:
            raise DeviceError(INVALID_ARGUMENT, "waypoint needs pose or joints")
        if "pose" in point:
            point["pose"] = finite_vector(point["pose"], 6, "pose")
        if "joints" in point:
            point["joints"] = finite_vector(point["joints"], self.dof, "joints")
        process = point.get("process", {})
        if not isinstance(process, dict):
            raise DeviceError(INVALID_ARGUMENT, "process must be an object")
        if "is_contact" in point and type(point["is_contact"]) is not bool:
            raise DeviceError(INVALID_ARGUMENT, "is_contact must be boolean")
        if "force" in process:
            finite_vector(process["force"], 3, "process.force")
        if process.get("valid"):
            for key in ("tcp_vel", "tcp_acc"):
                value = process.get(key, 0)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise DeviceError(INVALID_ARGUMENT, f"process.{key} must be positive")
        return point

    @staticmethod
    def _rotated_offset(pose, offset):
        rx, ry, rz = (math.radians(x) for x in pose[3:])
        sx, cx, sy, cy, sz, cz = math.sin(rx), math.cos(rx), math.sin(ry), math.cos(ry), math.sin(rz), math.cos(rz)
        rotation = ((cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
                    (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
                    (-sy, cy * sx, cy * cx))
        return [sum(a * b for a, b in zip(row, offset)) for row in rotation]

    def _tcp_target(self, point):
        target = deepcopy(point)
        if "pose" in target and target.get("pose_target", "tcp") == "flange":
            delta = self._rotated_offset(target["pose"], self._tools[self._tool])
            target["pose"][:3] = [a + b for a, b in zip(target["pose"][:3], delta)]
        return target

    def validate_path(self, points):
        """Validate and copy a trajectory without initializing or moving a robot."""
        if not isinstance(points, (list, tuple)) or not 1 <= len(points) <= 10000:
            raise DeviceError(INVALID_ARGUMENT, "path needs between 1 and 10000 points")
        return [self._point(point) for point in points]

    def move_path(self, points, duration_ms=None):
        with self._lock:
            admission_epoch = self._admission_epoch
        validated = self.validate_path(points)
        duration_ms = max(100, len(points) * 100) if duration_ms is None else duration_ms
        if isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)) or not math.isfinite(duration_ms) or not 0 < duration_ms <= 3600000:
            raise DeviceError(INVALID_ARGUMENT, "duration_ms must be positive and at most one hour")
        with self._lock:
            if admission_epoch != self._admission_epoch:
                raise DeviceError(CANCELLED, "motion was cancelled before submission")
            self._ready()
            if self._status in ("running", "paused"):
                raise DeviceError(BUSY, "robot already has an active motion")
            self._points = validated
            self._targets = [self._tcp_target(point) for point in validated]
            self._start_pose, self._start_joints = list(self._pose), list(self._joints)
            self._motion_id += 1
            self._elapsed, self._duration = 0.0, duration_ms / 1000
            self._progress, self._waypoint_index = 0.0, -1
            self._status = "running"
            return {"motion_id": self._motion_id}

    def move_joint(self, joints, relative=False, duration_ms=None):
        joints = finite_vector(joints, self.dof, "joints")
        if type(relative) is not bool:
            raise DeviceError(INVALID_ARGUMENT, "relative must be boolean")
        with self._lock:
            if relative:
                joints = [a + b for a, b in zip(self._joints, joints)]
            return self.move_path([{"joints": joints}], duration_ms)

    def move_home(self, preparatory=False, pose=None, joints=None, duration_ms=None):
        with self._lock:
            target = {"pose": self._preparatory_pose if preparatory else self._home_pose,
                      "joints": self._home_joints}
            if pose is not None:
                target["pose"] = pose
            if joints is not None:
                target["joints"] = joints
            return self.move_path([target], duration_ms)

    def advance(self, delta_time):
        if not isinstance(delta_time, (int, float)) or not math.isfinite(delta_time) or delta_time < 0:
            raise ValueError("delta_time must be finite and nonnegative")
        with self._lock:
            if self._status != "running":
                return
            self._elapsed = min(self._duration, self._elapsed + delta_time)
            self._progress = self._elapsed / self._duration
            position = self._progress * len(self._points)
            segment = min(int(position), len(self._points) - 1)
            alpha = min(1.0, position - segment)
            pose, joints = self._start_pose, self._start_joints
            for point in self._targets[:segment]:
                pose, joints = point.get("pose", pose), point.get("joints", joints)
            point = self._targets[segment]
            target_pose, target_joints = point.get("pose", pose), point.get("joints", joints)
            self._pose = [a + (b - a) * alpha for a, b in zip(pose, target_pose)]
            self._joints = [a + (b - a) * alpha for a, b in zip(joints, target_joints)]
            self._waypoint_index = len(self._points) - 1 if self._progress >= 1 else segment
            force = point.get("process", {}).get("force", [0, 0, 0]) if point.get("is_contact") else [0, 0, 0]
            self._wrench = [float(x) for x in force] + [0.0] * 3
            if self._progress >= 1:
                self._status = "completed"
                self._wrench = [0.0] * 6

    def stop(self, motion_id=None):
        with self._lock:
            if motion_id is not None and motion_id != self._motion_id:
                raise DeviceError(INVALID_ARGUMENT, "stale motion_id; current motion was not stopped")
            self._admission_epoch += 1
            if self._status in ("running", "paused"):
                self._status = "cancelled"
            self._wrench = [0.0] * 6
            return self.get_state()

    def pause(self):
        with self._lock:
            self._ready()
            if self._status != "running":
                raise DeviceError(BUSY, "robot is not running")
            self._status = "paused"
            return self.get_state()

    def resume(self):
        with self._lock:
            self._ready()
            if self._status != "paused":
                raise DeviceError(BUSY, "robot is not paused")
            self._status = "running"
            return self.get_state()

    def inject_fault(self, message="simulated robot fault"):
        if not isinstance(message, str) or not message:
            raise DeviceError(INVALID_ARGUMENT, "fault message must not be empty")
        with self._lock:
            self._admission_epoch += 1
            self._fault, self._status = str(message), "fault"
            self._wrench = [0.0] * 6
            return self.get_state()

    def clear_fault(self):
        with self._lock:
            if not self._connected:
                raise DeviceError(NOT_CONNECTED, "robot is not initialized")
            self._admission_epoch += 1
            self._fault, self._status = "", "idle"
            return self.get_state()

    def zero_ft_sensor(self):
        with self._lock:
            self._ready()
            self._wrench = [0.0] * 6
            return self.get_state()

    def set_tool(self, name, offset):
        offset = finite_vector(offset, 3, "tool offset")
        if not isinstance(name, str) or not name:
            raise DeviceError(INVALID_ARGUMENT, "tool name must not be empty")
        with self._lock:
            self._ready()
            if self._status in ("running", "paused"):
                raise DeviceError(BUSY, "cannot change tool during motion")
            previous_delta = self._rotated_offset(self._pose, self._tools[self._tool])
            flange_position = [a - b for a, b in zip(self._pose[:3], previous_delta)]
            self._tools[name], self._tool = offset, name
            delta = self._rotated_offset(self._pose, offset)
            self._pose[:3] = [a + b for a, b in zip(flange_position, delta)]
            return {"tool": name}

    def get_path(self):
        with self._lock:
            return deepcopy(self._points)

    def get_state(self):
        with self._lock:
            delta = self._rotated_offset(self._pose, self._tools[self._tool])
            flange = [a - b for a, b in zip(self._pose[:3], delta)] + self._pose[3:]
            return {"connected": self._connected, "brand": self.brand, "status": self._status,
                    "motion_id": self._motion_id, "pose": list(self._pose),
                    "flange_pose": flange, "pose_frame": self.pose_frame,
                    "joints": list(self._joints), "waypoint_index": self._waypoint_index,
                    "progress": self._progress, "wrench": list(self._wrench), "fault": self._fault,
                    "enabled": self._connected and not self._fault,
                    "moving": self._status == "running", "finished": self._status == "completed",
                    "program_state": "running" if self._status in ("running", "paused") else "stopped",
                    "tool": self._tool, "tools": list(self._tools)}

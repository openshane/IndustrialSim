"""Independent loopback HTTP service using Sensium device semantics.

The API models project device calls, not vendor SDK binary or wire protocols.
Each request gets a separate connection, so a waiting camera/I/O request never
holds the robot's Stop channel. UnInit/close wakes blocked device operations.
"""

import base64
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import threading
import time
from urllib.parse import urlparse

import cv2

from industrial_sim.core.errors import (
    BUSY, FAILURE, INVALID_ARGUMENT, NOT_CONNECTED, TIMEOUT, UNSUPPORTED, DeviceError,
)
from industrial_sim.core.config import CellConfig
from industrial_sim.devices.hikvision_camera import HikvisionCameraSimulator, PixelFormat, TriggerMode
from industrial_sim.devices.io import VirtualIO
from industrial_sim.devices.robot import VirtualRobot
from industrial_sim.devices.warning_light import VirtualWarningLight

logger = logging.getLogger(__name__)
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def validate_url(url):
    parsed = urlparse(url)
    if (parsed.scheme != "http" or parsed.hostname not in LOOPBACK_HOSTS
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("IndustrialSim URL must be a loopback http://host:port address")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid IndustrialSim port") from exc
    if port is None or not 1 <= port <= 65535:
        raise ValueError("IndustrialSim URL must include a valid explicit port")
    return parsed


class SensiumStation:
    """Owns one simulation clock and the configured virtual device instances."""

    def __init__(self, config=None):
        if config is not None and not isinstance(config, CellConfig):
            raise ValueError("config must be a validated CellConfig")
        self.config = None if config is None else CellConfig.from_dict(config.to_dict())
        self._lock = threading.RLock()
        self._drained = threading.Condition(self._lock)
        self._inflight = 0
        self._close_complete = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self.devices = {}
        self.kinds = {}
        self._capture_locks = {}
        self._init_defaults = {}
        if self.config is not None:
            for spec in self.config.devices:
                self._create(spec.function, spec.kind, spec.options)
            return
        for function, kind in (("CAMERA_0", "camera"), ("IO_0", "io"),
                               ("WARNING_LIGHT_0", "warning_light"),
                               ("FLEXIV_0", "flexiv"), ("DUCO_0", "duco"), ("KUKA_0", "kuka")):
            self._create(function, kind)

    def _create(self, function, kind, options=None):
        options = deepcopy(options or {})
        if kind == "camera":
            device = HikvisionCameraSimulator(function, seed=options.get("seed", 0))
            device.set_width(640)
            device.set_height(480)
            parameters = options.get("parameters", {})
            # TriggerSource is an independent camera attribute. Apply it first
            # so dictionary key order cannot change the requested trigger mode.
            if "TriggerSource" in parameters:
                self._camera_parameter(device, "TriggerSource", parameters["TriggerSource"])
            for name, value in parameters.items():
                if name != "TriggerSource":
                    self._camera_parameter(device, name, value)
            if options.get("image_files"):
                device.set_image_files(options["image_files"], loop=options.get("loop", True))
            self._capture_locks[function] = threading.Lock()
        elif kind == "io":
            device = VirtualIO(function, ports=options.get("ports", 2))
        elif kind == "warning_light":
            device = VirtualWarningLight(function)
        elif kind in VirtualRobot.PROFILES:
            device = VirtualRobot(kind, function)
            self._init_defaults[function] = options
        else:
            raise DeviceError(INVALID_ARGUMENT, "unknown device kind")
        self.devices[function], self.kinds[function] = device, kind
        return device

    def start(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("station is closed")
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="IndustrialSim-clock", daemon=True)
            self._thread.start()

    def _run(self):
        previous = time.monotonic()
        while not self._stop.wait(0.01):
            now = time.monotonic()
            with self._lock:
                robots = [d for d in self.devices.values() if isinstance(d, VirtualRobot)]
            for robot in robots:
                robot.advance(now - previous)
            previous = now

    def close(self):
        with self._lock:
            leader = not self._closed
            self._closed = True
            self._stop.set()
            devices = list(self.devices.values())
        if not leader:
            self._close_complete.wait()
            return
        errors = []
        try:
            # First interrupt waits, then drain accepted requests, then clean
            # again: a previously admitted Init/StartGrabbing may have finished
            # after the first pass. No user callback/join holds the station lock.
            for phase in range(2):
                for device in devices:
                    try:
                        if isinstance(device, HikvisionCameraSimulator):
                            device.disconnect()
                        else:
                            device.uninit()
                    except Exception as exc:
                        errors.append(exc)
                if phase == 0:
                    with self._drained:
                        self._drained.wait_for(lambda: self._inflight == 0)
            if self._thread is not None:
                self._thread.join(timeout=2)
                if self._thread.is_alive():
                    errors.append(RuntimeError("simulation clock did not stop"))
            if errors:
                raise RuntimeError("one or more virtual devices failed to stop") from errors[0]
        finally:
            self._close_complete.set()

    @staticmethod
    def _state(device):
        if isinstance(device, HikvisionCameraSimulator):
            return {"connected": device.is_connected(), "parameters": device.get_all_parameters()}
        return device.get_state()

    def snapshot(self):
        with self._lock:
            devices = list(self.devices.items())
        return {name: self._state(device) for name, device in devices}

    def configuration(self):
        return {"configured": self.config is not None,
                "config": None if self.config is None else self.config.to_dict()}

    def dispatch(self, request):
        with self._drained:
            if self._closed:
                raise DeviceError(NOT_CONNECTED, "station is closed")
            self._inflight += 1
        try:
            return self._dispatch(request)
        finally:
            with self._drained:
                self._inflight -= 1
                self._drained.notify_all()

    def _dispatch(self, request):
        if not isinstance(request, dict):
            raise DeviceError(INVALID_ARGUMENT, "request must be an object")
        function, command, args = request.get("device"), request.get("command"), request.get("args", {})
        if not isinstance(function, str) or not 1 <= len(function) <= 128:
            raise DeviceError(INVALID_ARGUMENT, "device must be a nonempty function name")
        if not isinstance(command, str) or not isinstance(args, dict):
            raise DeviceError(INVALID_ARGUMENT, "command must be a string and args an object")
        with self._lock:
            if self._closed:
                raise DeviceError(NOT_CONNECTED, "station is closed")
            kind = args.get("kind")
            if function not in self.devices:
                if self.config is not None or command != "Init" or kind is None:
                    raise DeviceError(NOT_CONNECTED, "unknown device; Init with an explicit kind first")
                self._create(function, kind)
            if kind is not None and kind != self.kinds[function]:
                raise DeviceError(INVALID_ARGUMENT, "device kind does not match its existing function")
            device = self.devices[function]
        if command in ("GetState", "GetDeviceStatus"):
            return self._state(device)
        if isinstance(device, HikvisionCameraSimulator):
            if command != "GetFrames":
                return self._camera(device, command, args)
            capture_lock = self._capture_locks[function]
            if not capture_lock.acquire(blocking=False):
                raise DeviceError(BUSY, "camera already has an active GetFrames request")
            try:
                return self._camera(device, command, args)
            finally:
                capture_lock.release()
        if command == "Init":
            if isinstance(device, VirtualRobot):
                values = {**self._init_defaults.get(function, {}),
                          **{k: v for k, v in args.items() if k != "kind"}}
                return device.init(**values)
            return device.init()
        if command == "UnInit":
            device.uninit()
            return {"connected": False}
        if isinstance(device, VirtualRobot):
            return self._robot(device, command, args)
        if isinstance(device, VirtualIO):
            if command == "Set":
                return device.set(args["index"], args["value"])
            if command == "Read":
                return device.read(args["index"])
            if command == "ReadReg":
                return device.read_reg(args["offset"])
            if command == "InjectInput":
                return device.inject_input(args["index"], args["value"])
            if command in ("ReadWait", "ReadRegWait"):
                key = "index" if command == "ReadWait" else "offset"
                return device.wait(**{key: args[key]}, after_sequence=args.get("after_sequence"),
                                   timeout_ms=args.get("timeout_ms", 0))
        if isinstance(device, VirtualWarningLight):
            states = {"AllOff": "off", "SetRunningState": "running", "SetWarningState": "warning", "SetErrorState": "error"}
            if command in states:
                return device.set_state(states[command], args.get("message", ""))
        raise DeviceError(UNSUPPORTED, f"unsupported command: {command}")

    @staticmethod
    def _robot(device, command, args):
        if command == "MoveJoint":
            return device.move_joint(args["joints"], args.get("relative", False), args.get("duration_ms"))
        if command in ("MovePath", "Polish"):
            return device.move_path(args["points"], args.get("duration_ms"))
        if command in ("MoveHome", "MovePreparatory"):
            return device.move_home(command == "MovePreparatory", args.get("pose"), args.get("joints"), args.get("duration_ms"))
        if command in ("Stop", "RobotStop"):
            return device.stop(args.get("motion_id"))
        if command == "Pause":
            return device.pause()
        if command == "Resume":
            return device.resume()
        if command == "InjectFault":
            return device.inject_fault(args.get("message", "simulated robot fault"))
        if command == "ClearFault":
            return device.clear_fault()
        if command == "ZeroFTSensor":
            return device.zero_ft_sensor()
        if command == "SetTool":
            return device.set_tool(args["name"], args.get("offset", [0, 0, 0]))
        if command == "GetToolList":
            return {"tools": device.get_state()["tools"]}
        if command == "GetPath":
            return {"points": device.get_path()}
        raise DeviceError(UNSUPPORTED, f"unsupported robot command: {command}")

    @staticmethod
    def _camera(device, command, args):
        if command == "Init":
            if "image_files" in args:
                device.set_image_files(args["image_files"], loop=args.get("loop", True))
            device.connect()
            return {"connected": device.is_connected()}
        if command == "UnInit":
            device.disconnect()
            return {"connected": False}
        if command == "StartGrabbing":
            if not device.start_grabbing(args.get("buffer_num", 10)):
                raise DeviceError(NOT_CONNECTED, "camera is not connected")
            return {"grabbing": True}
        if command == "StopGrabbing":
            device.stop_grabbing()
            return {"grabbing": False}
        if command == "TriggerCapture":
            if not device.trigger_capture():
                raise DeviceError(FAILURE, "camera did not accept the trigger")
            return {"accepted": True}
        if command == "InjectTimeout":
            enabled = args.get("enabled", True)
            if type(enabled) is not bool:
                raise ValueError("enabled must be boolean")
            device.inject_timeout(enabled)
            return {"enabled": enabled}
        if command == "SetImageFiles":
            device.set_image_files(args["paths"], loop=args.get("loop", True))
            return {}
        if command == "SetParameter":
            return SensiumStation._camera_parameter(device, args["name"], args["value"])
        if command == "GetFrames":
            count, timeout = args.get("frame_count", 1), args.get("timeout_ms", 1000)
            if type(count) is not int or not 1 <= count <= 64:
                raise ValueError("frame_count must be between 1 and 64")
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 <= timeout <= 60000:
                raise ValueError("camera timeout_ms must be between 0 and 60000")
            auto_trigger = args.get("auto_trigger", False)
            if type(auto_trigger) is not bool:
                raise ValueError("auto_trigger must be boolean")
            if auto_trigger:
                device.clear_frames()
            deadline = time.monotonic() + timeout / 1000
            frames = []
            for _ in range(count):
                if auto_trigger:
                    if not device.trigger_capture():
                        raise DeviceError(FAILURE, "camera trigger queue is full or camera is stopped")
                frames.extend(device.get_frames(1, max(0, deadline - time.monotonic())))
            result = []
            for frame in frames:
                image = frame.data
                if frame.pixel_format == PixelFormat.RGB8:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                ok, encoded = cv2.imencode(".png", image)
                if not ok:
                    raise DeviceError(FAILURE, "failed to encode camera frame")
                result.append({"width": frame.width, "height": frame.height,
                               "pixel_format": frame.pixel_format.value, "frame_id": frame.frame_id,
                               "timestamp": frame.timestamp,
                               "png_base64": base64.b64encode(encoded).decode("ascii")})
            return {"frames": result}
        raise DeviceError(UNSUPPORTED, f"unsupported camera command: {command}")

    @staticmethod
    def _camera_parameter(device, name, value):
        setters = {"Width": device.set_width, "Height": device.set_height,
                   "ExposureTime": device.set_exposure_time, "Gain": device.set_gain,
                   "AcquisitionFrameRate": device.set_frame_rate, "FrameRate": device.set_frame_rate}
        if name in setters:
            setters[name](value)
        elif name == "PixelFormat":
            device.set_pixel_format(PixelFormat(value))
        elif name == "TriggerMode":
            if value in ("Off", "Continuous"):
                device.set_trigger_mode(TriggerMode.CONTINUOUS)
            elif value == "Software":
                device.set_trigger_mode(TriggerMode.SOFTWARE)
            elif value == "On":
                source = device.get_all_parameters()["TriggerSource"]
                device.set_trigger_mode(TriggerMode.SOFTWARE if source == "Software" else TriggerMode.HARDWARE)
            elif value == "Hardware":
                device.set_trigger_mode(TriggerMode.HARDWARE)
            else:
                raise ValueError("invalid TriggerMode")
        elif name == "TriggerSource":
            device.set_trigger_source(value)
        elif name == "AcquisitionMode" and value == "Continuous":
            pass
        else:
            raise DeviceError(UNSUPPORTED, f"camera parameter is not modeled: {name}")
        return {"parameters": device.get_all_parameters()}


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = False

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(5)
        return connection, address


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)

    def _reply(self, value, status=200):
        payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def do_GET(self):
        if self.path == "/health":
            self._reply({"service": "IndustrialSim", "protocol_version": 1})
        elif self.path == "/v1/state":
            self._reply(self.server.station.snapshot())
        elif self.path == "/v1/config":
            self._reply(self.server.station.configuration())
        else:
            self._reply({"code": INVALID_ARGUMENT, "message": "unknown endpoint", "data": {}}, 404)

    def do_POST(self):
        if self.path != "/v1/command":
            self._reply({"code": INVALID_ARGUMENT, "message": "unknown endpoint", "data": {}}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 2 * 1024 * 1024:
                raise ValueError("request body must be between 1 byte and 2 MiB")
            request = json.loads(self.rfile.read(length).decode("utf-8"),
                                 parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid number {value}")))
            result = self.server.station.dispatch(request)
            response = {"code": 0, "message": "", "data": result}
        except DeviceError as exc:
            response = {"code": exc.code, "message": str(exc), "data": {}}
        except TimeoutError as exc:
            response = {"code": TIMEOUT, "message": str(exc), "data": {"value": 0, "frames": []}}
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            response = {"code": INVALID_ARGUMENT, "message": str(exc), "data": {}}
        except RuntimeError as exc:
            response = {"code": FAILURE, "message": str(exc), "data": {}}
        except Exception:
            logger.exception("device command failed")
            response = {"code": FAILURE, "message": "internal simulation error", "data": {}}
        self._reply(response)


class SimulatorServer:
    def __init__(self, host="127.0.0.1", port=0, station=None, config=None):
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("server must bind to IPv4 loopback")
        if station is not None and config is not None:
            raise ValueError("specify station or config, not both")
        self.station = station or SensiumStation(config=config)
        self._server = _Server((host, port), _Handler)
        self._server.station = self.station
        self._thread = None
        self._closed = False
        self._close_lock = threading.Lock()
        self._close_complete = threading.Event()

    @property
    def url(self):
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self):
        if self._closed:
            raise RuntimeError("server is closed")
        if self._thread is None:
            self.station.start()
            self._thread = threading.Thread(target=self._server.serve_forever, name="IndustrialSim-http", daemon=True)
            self._thread.start()
        return self

    def close(self):
        with self._close_lock:
            leader = not self._closed
            self._closed = True
        if not leader:
            self._close_complete.wait()
            return
        try:
            try:
                self.station.close()
            finally:
                if self._thread is not None:
                    self._server.shutdown()
                    self._thread.join(timeout=3)
                self._server.server_close()
        finally:
            self._close_complete.set()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

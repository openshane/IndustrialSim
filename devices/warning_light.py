"""Semantic model of GrindingTool's serial warning light."""

import threading

from industrial_sim.core.errors import NOT_CONNECTED, DeviceError


class VirtualWarningLight:
    def __init__(self, function="WARNING_LIGHT_0"):
        self.function = function
        self._lock = threading.RLock()
        self._connected = False
        self._state = "off"
        self._message = ""

    def init(self):
        with self._lock:
            self._connected = True
            self._state = "running"
            return self.get_state()

    def uninit(self):
        with self._lock:
            self._state = "off"
            self._message = ""
            self._connected = False

    def set_state(self, state, message=""):
        if state not in ("off", "running", "warning", "error"):
            raise ValueError("invalid warning light state")
        with self._lock:
            if not self._connected:
                raise DeviceError(NOT_CONNECTED, "warning light is not initialized")
            self._state, self._message = state, str(message)
            return self.get_state()

    def get_state(self):
        with self._lock:
            color = {"running": "green", "warning": "yellow", "error": "red"}.get(self._state)
            hz = {"warning": 1, "error": 2}.get(self._state, 0)
            return {"connected": self._connected, "state": self._state,
                    "colors": [] if color is None else [color], "buzzer": hz > 0,
                    "buzzer_hz": hz, "message": self._message}

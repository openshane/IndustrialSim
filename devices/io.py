"""Digital I/O with the edge-wait contract used by GrindingTool."""

from collections import deque
import threading
import time

from industrial_sim.core.errors import CANCELLED, INVALID_ARGUMENT, NOT_CONNECTED, DeviceError


class VirtualIO:
    def __init__(self, function="IO_0", ports=2):
        if not isinstance(ports, int) or not 1 <= ports <= 32:
            raise ValueError("ports must be between 1 and 32")
        self.function = function
        self._inputs = [0] * (ports * 8)
        self._outputs = [0] * (ports * 8)
        self._condition = threading.Condition(threading.RLock())
        self._connected = False
        self._generation = 0
        self._sequence = 0
        self._events = deque(maxlen=4096)

    def init(self):
        with self._condition:
            if not self._connected:
                self._connected = True
                self._generation += 1
            return self.get_state()

    def uninit(self):
        with self._condition:
            self._connected = False
            self._generation += 1
            self._outputs[:] = [0] * len(self._outputs)
            self._condition.notify_all()

    def _require_connected(self):
        if not self._connected:
            raise DeviceError(NOT_CONNECTED, "I/O is not initialized")

    def _index(self, index):
        if type(index) is not int or not 0 <= index < len(self._inputs):
            raise DeviceError(INVALID_ARGUMENT, "I/O index is out of range")
        return index

    def _offset(self, offset):
        if type(offset) is not int or not 0 <= offset < len(self._inputs) // 8:
            raise DeviceError(INVALID_ARGUMENT, "I/O port offset is out of range")
        return offset

    @staticmethod
    def _bit(value):
        if value not in (0, 1) or not isinstance(value, (int, bool)):
            raise DeviceError(INVALID_ARGUMENT, "digital signal must be 0 or 1")
        return int(value)

    @staticmethod
    def _byte(bits, offset):
        return sum(bits[offset * 8 + bit] << bit for bit in range(8))

    def get_state(self):
        with self._condition:
            return {"connected": self._connected, "inputs": list(self._inputs),
                    "outputs": list(self._outputs), "sequence": self._sequence}

    def set(self, index, value):
        with self._condition:
            self._require_connected()
            self._outputs[self._index(index)] = self._bit(value)
            return {"value": self._outputs[index]}

    def read(self, index):
        with self._condition:
            self._require_connected()
            return {"value": self._inputs[self._index(index)], "sequence": self._sequence}

    def read_reg(self, offset):
        with self._condition:
            self._require_connected()
            return {"value": self._byte(self._inputs, self._offset(offset)),
                    "sequence": self._sequence}

    def inject_input(self, index, value):
        with self._condition:
            self._require_connected()
            index, value = self._index(index), self._bit(value)
            if self._inputs[index] != value:
                self._inputs[index] = value
                self._sequence += 1
                self._events.append((self._sequence, index, value,
                                     self._byte(self._inputs, index // 8)))
                self._condition.notify_all()
            return {"value": value, "sequence": self._sequence}

    def wait(self, *, index=None, offset=None, after_sequence=None, timeout_ms=0):
        """Wait for the next matching edge, including falling edges.

        An explicit cursor lets a caller safely arm a wait before another thread
        injects a signal. A stale cursor fails instead of silently losing edges.
        """
        if type(timeout_ms) not in (int, float) or not 0 <= timeout_ms <= 86400000:
            raise DeviceError(INVALID_ARGUMENT, "timeout_ms must be between 0 and 86400000")
        with self._condition:
            self._require_connected()
            if (index is None) == (offset is None):
                raise DeviceError(INVALID_ARGUMENT, "specify either index or offset")
            if index is not None:
                index = self._index(index)
            else:
                offset = self._offset(offset)
            cursor = self._sequence if after_sequence is None else after_sequence
            if type(cursor) is not int or not 0 <= cursor <= self._sequence:
                raise DeviceError(INVALID_ARGUMENT, "invalid event cursor")
            generation = self._generation
            deadline = time.monotonic() + timeout_ms / 1000 if timeout_ms else None
            while True:
                if generation != self._generation or not self._connected:
                    raise DeviceError(CANCELLED, "I/O wait cancelled by UnInit")
                if self._events and cursor < self._events[0][0] - 1:
                    raise DeviceError(INVALID_ARGUMENT, "event cursor expired; read GetState again")
                for sequence, bit, value, port_value in self._events:
                    if sequence > cursor and (bit == index if index is not None else bit // 8 == offset):
                        return {"value": value if index is not None else port_value,
                                "sequence": sequence}
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("no matching I/O edge before timeout")
                self._condition.wait(remaining)

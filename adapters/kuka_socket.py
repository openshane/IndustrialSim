"""A virtual KUKA protocol peer for GrindingTool's existing XML/TCP workflow.

GrindingTool listens; this peer connects and requests a configured Function.
Trajectory batches are acknowledged only after the supplied consumer returns.
This implements the project's application protocol, not KUKA RSI/EKI itself.
"""

from dataclasses import dataclass, field
import math
import re
import socket
import time
import xml.etree.ElementTree as ET

from industrial_sim.core.errors import FAULT, INVALID_ARGUMENT, DeviceError

ACK = b"<Robot><Status>TrackProcessEnd</Status></Robot>"


@dataclass
class KukaExchange:
    packets: list = field(default_factory=list)
    points: list = field(default_factory=list)
    acknowledgements: int = 0


class KukaSocketSimulator:
    def __init__(self, host="127.0.0.1", port=5000, timeout=10):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("virtual KUKA may only connect to a loopback GrindingTool server")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("invalid socket port")
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.host, self.port, self.timeout = host, port, timeout

    @staticmethod
    def _request(function, fields):
        if not isinstance(function, str) or not function:
            raise ValueError("a configured Function name is required")
        root = ET.Element("Robot")
        ET.SubElement(root, "Function").text = function
        for name, value in (fields or {}).items():
            if name == "Function" or not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", name):
                raise ValueError("invalid request field")
            ET.SubElement(root, name).text = str(value)
        request = ET.tostring(root, encoding="utf-8", short_empty_elements=False)
        if len(request) > 1023:
            raise ValueError("GrindingTool's current request receiver accepts at most 1023 bytes")
        return request

    def exchange(self, function, fields=None, on_batch=None):
        request = self._request(function, fields)
        result = KukaExchange()
        deadline = time.monotonic() + self.timeout
        buffer = bytearray()
        totals = None
        float_count = integer_count = 0
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as connection:
            # The receiver uses one receiveBytes call; send one complete XML
            # document without a newline. TCP may still fragment it: that is a
            # limitation of the current native receiver, not a framing promise.
            connection.sendall(request)
            while True:
                terminator = b"</Computer>"
                while terminator not in buffer:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("KUKA workflow response timed out")
                    connection.settimeout(remaining)
                    data = connection.recv(65536)
                    if not data:
                        raise ConnectionError("GrindingTool closed an incomplete XML response")
                    buffer.extend(data)
                    if len(buffer) > 8 * 1024 * 1024:
                        raise ValueError("KUKA response exceeds 8 MiB")
                end = buffer.index(terminator) + len(terminator)
                document = bytes(buffer[:end])
                del buffer[:end]
                root = ET.fromstring(document)
                if root.tag != "Computer":
                    raise ValueError("expected a Computer response")
                code = int(root.findtext("Code", "0"))
                if code != 0:
                    raise DeviceError(FAULT, f"GrindingTool workflow returned Code={code}")
                result.packets.append(document.decode("utf-8"))
                floats = [float(node.text) for node in root.findall("MD0")]
                integers = [int(node.text) for node in root.findall("MD1")]
                if any(not math.isfinite(value) for value in floats):
                    raise DeviceError(INVALID_ARGUMENT, "nonfinite KUKA trajectory value")
                if root.find("S0") is None or root.find("S1") is None:
                    # Non-trajectory replies (location, status, errors) consist
                    # of one packet and must not receive a trajectory ACK.
                    if on_batch is not None:
                        on_batch([])
                    return result
                packet_totals = (int(root.findtext("S0")), int(root.findtext("S1")))
                if any(value < 0 for value in packet_totals):
                    raise ValueError("negative trajectory counts")
                if totals is None:
                    totals = packet_totals
                elif totals != packet_totals:
                    raise ValueError("trajectory totals changed between batches")
                if totals == (0, 0):
                    return result
                if totals[1] <= 0 or totals[1] % 2:
                    raise ValueError("expected two configuration integers per trajectory point")
                point_count = totals[1] // 2
                if totals[0] % point_count:
                    raise ValueError("invalid trajectory float count")
                stride = totals[0] // point_count
                if stride not in (6, 7, 8) or len(floats) % stride or len(integers) != len(floats) // stride * 2:
                    raise ValueError("invalid KUKA trajectory batch shape")
                remaining_points = point_count - len(result.points)
                count = min(len(floats) // stride, remaining_points)
                if count == 0:
                    raise ValueError("empty trajectory batch before completion")
                batch = []
                for index in range(count):
                    values = floats[index * stride:(index + 1) * stride]
                    batch.append({"pose": values[:3] + [values[5], values[4], values[3]],
                                  "reference_frame": "world", "pose_target": "tcp",
                                  "s": integers[index * 2], "t": integers[index * 2 + 1],
                                  "ext_axes": values[6:]})
                if on_batch is not None:
                    on_batch(batch)
                result.points.extend(batch)
                float_count += count * stride
                integer_count += count * 2
                if (float_count, integer_count) == totals:
                    return result
                # No extra newline: the current native sender compares this
                # exact string before sending the next batch.
                connection.sendall(ACK)
                result.acknowledgements += 1

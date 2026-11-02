"""Small dependency-free HTTP client for local virtual device calls."""

import base64
import http.client
import json
import socket

import cv2
import numpy as np

from industrial_sim.adapters.sensium import validate_url
from industrial_sim.core.errors import FAILURE, NOT_CONNECTED, TIMEOUT, DeviceError


class IndustrialSimClient:
    def __init__(self, url, timeout=3.0):
        self._url = validate_url(url)
        self.timeout = timeout

    def _request(self, method, path, body=None, timeout=None):
        connection = http.client.HTTPConnection(self._url.hostname, self._url.port,
                                                timeout=self.timeout if timeout is None else timeout)
        try:
            payload = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
            connection.request(method, path, payload, {"Content-Type": "application/json"})
            response = connection.getresponse()
            content = response.read(32 * 1024 * 1024 + 1)
            if response.status != 200 or len(content) > 32 * 1024 * 1024:
                raise DeviceError(FAILURE, f"invalid simulation HTTP response ({response.status})")
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError("expected a JSON object")
            return result
        except (socket.timeout, TimeoutError) as exc:
            raise DeviceError(TIMEOUT, "IndustrialSim HTTP request timed out") from exc
        except (OSError, http.client.HTTPException) as exc:
            raise DeviceError(NOT_CONNECTED, "IndustrialSim service is unavailable") from exc
        except (ValueError, UnicodeError) as exc:
            raise DeviceError(FAILURE, "invalid IndustrialSim JSON response") from exc
        finally:
            connection.close()

    def health(self):
        result = self._request("GET", "/health")
        if result.get("service") != "IndustrialSim" or result.get("protocol_version") != 1:
            raise DeviceError(FAILURE, "incompatible IndustrialSim service")
        return result

    def call(self, device, command, *, timeout=None, **args):
        result = self._request("POST", "/v1/command", {"device": device, "command": command, "args": args}, timeout)
        if type(result.get("code")) is not int or not isinstance(result.get("data"), dict):
            raise DeviceError(FAILURE, "invalid device response contract")
        if result["code"] != 0:
            raise DeviceError(result["code"], str(result.get("message", "device operation failed")))
        return result["data"]

    def state(self, device):
        return self.call(device, "GetState")

    def configuration(self):
        result = self._request("GET", "/v1/config")
        if type(result.get("configured")) is not bool or (result["configured"] and not isinstance(result.get("config"), dict)):
            raise DeviceError(FAILURE, "invalid service configuration response")
        return result

    def get_frames(self, device="CAMERA_0", frame_count=1, timeout_ms=1000, auto_trigger=True):
        data = self.call(device, "GetFrames", frame_count=frame_count, timeout_ms=timeout_ms,
                         auto_trigger=auto_trigger, timeout=timeout_ms / 1000 + 2)
        frames = []
        for metadata in data["frames"]:
            encoded = base64.b64decode(metadata["png_base64"], validate=True)
            image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None or image.shape[:2] != (metadata["height"], metadata["width"]):
                raise DeviceError(FAILURE, "invalid camera PNG or dimensions")
            # PNG is encoded with OpenCV's BGR convention on the server. Python
            # callers receive RGB8 in RGB order, matching FrameData/metadata;
            # the native adapter decodes the same PNG directly into a BGR Mat.
            if metadata["pixel_format"] == "RGB8":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(({k: v for k, v in metadata.items() if k != "png_base64"}, image))
        return frames

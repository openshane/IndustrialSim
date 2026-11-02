"""Transport-independent failures for virtual devices."""

SUCCESS = 0
FAILURE = -1
INVALID_ARGUMENT = -2
NOT_CONNECTED = -3
BUSY = -4
TIMEOUT = -5
CANCELLED = -6
FAULT = -7
UNSUPPORTED = -8


class DeviceError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code

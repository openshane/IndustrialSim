from contextlib import contextmanager
import socket
import threading
import xml.etree.ElementTree as ET

import pytest

from industrial_sim.adapters.kuka_socket import ACK, KukaSocketSimulator
from industrial_sim.core.errors import DeviceError


@contextmanager
def fake_grindingtool(handler):
    errors = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(2)

        def run():
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(2)
                    handler(connection)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run, name="test-kuka-native-peer")
        thread.start()
        try:
            yield listener.getsockname()[1]
        finally:
            thread.join(3)
            assert not thread.is_alive(), "test TCP peer did not exit"
            if errors:
                raise errors[0]


def packet(points, total_points, axes=0, padding=0):
    data = []
    for pose, s, t in points + [([0] * (6 + axes), 0, 0)] * padding:
        # Wire order x,y,z,rz,ry,rx is intentionally different from the API.
        wire = pose[:3] + [pose[5], pose[4], pose[3]]
        data.extend(f"<MD0>{value}</MD0>" for value in wire)
        data.extend((f"<MD1>{s}</MD1>", f"<MD1>{t}</MD1>"))
        data.extend(f"<MD0>{value}</MD0>" for value in pose[6:])
    return ("<Computer>" + "".join(data) + f"<S0>{total_points * (6 + axes)}</S0>"
            f"<S1>{total_points * 2}</S1><Code>0</Code></Computer>").encode()


def test_batches_ack_after_consumer_and_no_ack_for_final_batch():
    consumed = []

    def handler(connection):
        request = connection.recv(1024)
        root = ET.fromstring(request)
        assert root.tag == "Robot"
        assert root.findtext("Function") == "Grinding"
        assert root.findtext("Part") == "A&B"
        first = packet([([1, 2, 3, 4, 5, 6], 8, 9)], 2)
        # Exercise fragmented responses rather than relying on recv boundaries.
        connection.sendall(first[:17])
        connection.sendall(first[17:])
        assert connection.recv(1024) == ACK
        assert consumed == [1]
        connection.sendall(packet([([10, 20, 30, 40, 50, 60], 2, 3)], 2))
        assert connection.recv(1024) == b""

    with fake_grindingtool(handler) as port:
        exchange = KukaSocketSimulator(port=port).exchange("Grinding", {"Part": "A&B"},
                                                           lambda points: consumed.append(len(points)))
    assert exchange.acknowledgements == 1
    assert exchange.points[0]["pose"] == [1, 2, 3, 4, 5, 6]
    assert exchange.points[1]["pose"] == [10, 20, 30, 40, 50, 60]
    assert exchange.points[0]["s"] == 8 and exchange.points[0]["t"] == 9


@pytest.mark.parametrize("axes", [1, 2])
def test_external_axes_order_and_padded_last_batch(axes):
    pose = [1, 2, 3, 4, 5, 6] + list(range(7, 7 + axes))

    def handler(connection):
        connection.recv(1024)
        connection.sendall(packet([(pose, 1, 2)], 1, axes=axes, padding=2 if axes == 2 else 0))
        assert connection.recv(1024) == b""

    with fake_grindingtool(handler) as port:
        exchange = KukaSocketSimulator(port=port).exchange("Grinding")
    assert len(exchange.points) == 1
    assert exchange.points[0]["pose"] == pose[:6]
    assert exchange.points[0]["ext_axes"] == pose[6:]


def test_nontrajectory_reply_needs_no_ack():
    def handler(connection):
        connection.recv(1024)
        connection.sendall(b"<Computer><MD0>1.5</MD0><S>1</S><Code>0</Code></Computer>")
        assert connection.recv(1024) == b""

    with fake_grindingtool(handler) as port:
        exchange = KukaSocketSimulator(port=port).exchange("Locate")
    assert len(exchange.packets) == 1
    assert exchange.acknowledgements == 0


def test_native_workflow_error_is_not_success():
    def handler(connection):
        connection.recv(1024)
        connection.sendall(b"<Computer><Code>4004</Code></Computer>")

    with fake_grindingtool(handler) as port:
        with pytest.raises(DeviceError, match="4004"):
            KukaSocketSimulator(port=port).exchange("Grinding")


def test_truncated_xml_fails():
    def handler(connection):
        connection.recv(1024)
        connection.sendall(b"<Computer><Code>0</Code>")

    with fake_grindingtool(handler) as port:
        with pytest.raises(ConnectionError, match="incomplete"):
            KukaSocketSimulator(port=port).exchange("Grinding")


def test_inconsistent_totals_fail_before_acknowledging_bad_batch():
    def handler(connection):
        connection.recv(1024)
        connection.sendall(packet([([0] * 6, 0, 0)], 3))
        assert connection.recv(1024) == ACK
        connection.sendall(packet([([0] * 6, 0, 0)], 2))
        assert connection.recv(1024) == b""

    with fake_grindingtool(handler) as port:
        with pytest.raises(ValueError, match="totals changed"):
            KukaSocketSimulator(port=port).exchange("Grinding")


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com"])
def test_peer_rejects_nonloopback_targets(host):
    with pytest.raises(ValueError, match="loopback"):
        KukaSocketSimulator(host=host)


def test_request_validation_and_no_newline():
    request = KukaSocketSimulator._request("Grinding", {"Part": "A&B"})
    assert request.endswith(b"</Robot>")
    assert b"\n" not in request
    assert ET.fromstring(request).findtext("Part") == "A&B"
    with pytest.raises(ValueError, match="1023"):
        KukaSocketSimulator._request("Grinding", {"Part": "x" * 2000})
    with pytest.raises(ValueError, match="field"):
        KukaSocketSimulator._request("Grinding", {"1bad": 2})

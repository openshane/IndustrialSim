"""配置入口、交叉引用与直接构造的校验，不启动设备或网络。"""

from copy import deepcopy
import json
from pathlib import Path
import re

import cv2
import numpy as np
import pytest

from industrial_sim.core.config import (
    CellConfig,
    ConfigurationError,
    CycleSpec,
    DeviceSpec,
    load_cell_config,
)


@pytest.fixture
def document():
    return {
        "version": 1,
        "devices": [
            {"function": "CAM_A", "kind": "camera", "seed": 42,
             "parameters": {"Width": 12, "Height": 8, "PixelFormat": "RGB8", "TriggerMode": "Software"}},
            {"function": "PLC_A", "kind": "io", "ports": 2},
            {"function": "ARM_A", "kind": "duco", "home_pose": [0] * 6,
             "home_joints": [0] * 6, "preparatory_pose": [0] * 6},
            {"function": "LIGHT_A", "kind": "warning_light"},
        ],
        "cycle": {
            "io": "PLC_A", "camera": "CAM_A", "robot": "ARM_A", "light": "LIGHT_A",
            "arrival_input": 9, "complete_output": 8, "running_output": 9, "fault_output": 10,
            "points": [{"pose": [1, 2, 3, 4, 5, 6]}], "duration_ms": 80,
            "arrival_timeout_ms": 1000, "camera_timeout_ms": 1000, "motion_timeout_ms": 2000,
        },
    }


def replace_field(document, path, value):
    current = document
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def test_valid_config_preserves_custom_names_points_and_process_metadata(document):
    point = document["cycle"]["points"][0]
    point.update({"reference_frame": "world", "pose_target": "tcp", "is_contact": True,
                  "process": {"valid": True, "tcp_vel": 10, "tcp_acc": 20, "force": [0, 0, -2]}})
    config = CellConfig.from_dict(document)
    assert isinstance(config.devices, tuple)
    assert config.find_device("CAM_A").options["seed"] == 42
    assert config.find_device("ARM_A").kind == "duco"
    assert config.cycle.arrival_input == 9
    assert config.cycle.running_output == 9  # DI and DO are separate address spaces.
    saved = config.to_dict()
    assert saved["cycle"]["points"][0] == point
    assert saved["devices"][0]["parameters"] == document["devices"][0]["parameters"]
    assert CellConfig.from_dict(saved).to_dict() == saved
    json.dumps(saved, allow_nan=False)


def test_parser_and_to_dict_do_not_share_mutable_documents(document):
    config = CellConfig.from_dict(document)
    document["devices"][0]["parameters"]["Width"] = 999
    document["cycle"]["points"][0]["pose"][0] = 999
    saved = config.to_dict()
    assert saved["devices"][0]["parameters"]["Width"] == 12
    assert saved["cycle"]["points"][0]["pose"][0] == 1
    saved["devices"][0]["parameters"]["Width"] = -1
    saved["cycle"]["points"][0]["pose"].clear()
    assert config.to_dict()["devices"][0]["parameters"]["Width"] == 12
    assert len(config.to_dict()["cycle"]["points"][0]["pose"]) == 6


def test_cycle_can_be_omitted_for_a_device_only_service(document):
    del document["cycle"]
    config = CellConfig.from_dict(document)
    assert config.cycle is None
    assert {device.function for device in config.devices} == {"CAM_A", "PLC_A", "ARM_A", "LIGHT_A"}
    assert CellConfig.from_dict(config.to_dict()).cycle is None


@pytest.mark.parametrize("kind,dof", [("flexiv", 7), ("duco", 6), ("kuka", 6)])
def test_robot_home_and_path_joint_dimensions_match_profile(document, kind, dof):
    document["devices"][2]["kind"] = kind
    document["devices"][2]["home_joints"] = [0] * dof
    document["cycle"]["points"] = [{"joints": [1] * dof}]
    assert CellConfig.from_dict(document).find_device("ARM_A").kind == kind
    document["cycle"]["points"] = [{"joints": [1] * (dof - 1)}]
    with pytest.raises(ConfigurationError, match="joints"):
        CellConfig.from_dict(document)


@pytest.mark.parametrize("path,value,field", [
    (("version",), True, "version"),
    (("version",), 1.0, "version"),
    (("version",), 2, "version"),
    (("version",), "1", "version"),
    (("devices",), {}, "devices"),
    (("devices", 0, "function"), "", "function"),
    (("devices", 0, "function"), 7, "function"),
    (("devices", 0, "kind"), "unknown", "kind"),
    (("devices", 0, "seed"), True, "seed"),
    (("devices", 0, "seed"), -1, "seed"),
    (("devices", 0, "parameters", "Width"), True, "Width"),
    (("devices", 0, "parameters", "Width"), 0, "Width"),
    (("devices", 0, "parameters", "Height"), 1.5, "Height"),
    (("devices", 0, "parameters", "PixelFormat"), "bad", "PixelFormat"),
    (("devices", 0, "parameters", "TriggerMode"), "bad", "TriggerMode"),
    (("devices", 0, "parameters", "FrameRate"), float("nan"), "FrameRate"),
    (("devices", 0, "parameters", "Gain"), float("inf"), "Gain"),
    (("devices", 0, "parameters", "ExposureTime"), True, "ExposureTime"),
    (("devices", 0, "image_files"), "one.png", "image_files"),
    (("devices", 0, "loop"), "false", "loop"),
    (("devices", 1, "ports"), True, "ports"),
    (("devices", 1, "ports"), 0, "ports"),
    (("devices", 2, "home_pose"), [0] * 5, "home_pose"),
    (("devices", 2, "home_joints"), [0] * 7, "home_joints"),
    (("devices", 2, "preparatory_pose"), [float("inf")] * 6, "preparatory_pose"),
    (("cycle", "arrival_input"), True, "arrival_input"),
    (("cycle", "complete_output"), -1, "complete_output"),
    (("cycle", "running_output"), 1.5, "running_output"),
    (("cycle", "duration_ms"), True, "duration_ms"),
    (("cycle", "duration_ms"), float("nan"), "duration_ms"),
    (("cycle", "camera_timeout_ms"), -1, "camera_timeout_ms"),
    (("cycle", "arrival_timeout_ms"), True, "arrival_timeout_ms"),
    (("cycle", "motion_timeout_ms"), float("inf"), "motion_timeout_ms"),
    (("cycle", "frame_count"), True, "frame_count"),
    (("cycle", "frame_count"), 65, "frame_count"),
    (("cycle", "points"), [], "points"),
    (("cycle", "points"), [{}], "points"),
    (("cycle", "points"), [{"pose": [0] * 5}], "pose"),
    (("cycle", "points"), [{"pose": [float("nan")] * 6}], "pose"),
    (("cycle", "points"), [{"pose": [0] * 6, "is_contact": 1}], "is_contact"),
    (("cycle", "points"), [{"pose": [0] * 6, "process": {"force": ["bad", 0, 0]}}], "force"),
])
def test_invalid_fields_report_configuration_error_with_location(document, path, value, field):
    replace_field(document, path, value)
    with pytest.raises(ConfigurationError) as caught:
        CellConfig.from_dict(document)
    assert re.search(rf"\b{re.escape(field)}\b", str(caught.value))


@pytest.mark.parametrize("path", [(), ("devices", 0), ("devices", 0, "parameters"), ("cycle",)])
def test_unknown_fields_are_rejected_instead_of_ignored(document, path):
    current = document
    for part in path:
        current = current[part]
    current["misspelled_field"] = 123
    with pytest.raises(ConfigurationError, match="misspelled_field"):
        CellConfig.from_dict(document)


def test_duplicate_function_is_rejected(document):
    document["devices"].append(deepcopy(document["devices"][0]))
    with pytest.raises(ConfigurationError, match="function"):
        CellConfig.from_dict(document)


@pytest.mark.parametrize("field,reference", [
    ("io", "MISSING"), ("camera", "MISSING"), ("robot", "MISSING"), ("light", "MISSING"),
    ("io", "CAM_A"), ("camera", "PLC_A"), ("robot", "LIGHT_A"), ("light", "ARM_A"),
])
def test_cycle_references_must_exist_and_match_device_kind(document, field, reference):
    document["cycle"][field] = reference
    with pytest.raises(ConfigurationError) as caught:
        CellConfig.from_dict(document)
    assert re.search(rf"\b{re.escape(field)}\b", str(caught.value))


def test_output_mappings_must_be_distinct_but_can_share_input_number(document):
    assert CellConfig.from_dict(document).cycle.arrival_input == 9
    document["cycle"]["fault_output"] = document["cycle"]["complete_output"]
    with pytest.raises(ConfigurationError, match="output"):
        CellConfig.from_dict(document)


@pytest.mark.parametrize("field", ["arrival_input", "complete_output", "running_output", "fault_output"])
def test_io_addresses_respect_configured_port_count(document, field):
    document["devices"][1]["ports"] = 1
    document["cycle"].update(arrival_input=0, complete_output=0, running_output=1, fault_output=2)
    document["cycle"][field] = 8
    with pytest.raises(ConfigurationError) as caught:
        CellConfig.from_dict(document)
    assert field in str(caught.value)


def test_relative_images_resolve_against_configuration_directory(document, tmp_path, monkeypatch):
    directory = tmp_path / "配置"
    directory.mkdir()
    encoded, image = cv2.imencode(".png", np.full((8, 12, 3), (7, 19, 231), np.uint8))
    assert encoded
    image.tofile(directory / "sample.png")
    document["devices"][0]["image_files"] = ["sample.png"]
    document["devices"][0]["loop"] = False
    path = directory / "cell.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    elsewhere = tmp_path / "other-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_cell_config(path)
    camera = config.find_device("CAM_A")
    assert Path(camera.options["image_files"][0]) == (directory / "sample.png").resolve()
    assert camera.options["loop"] is False
    direct = CellConfig.from_dict(document, base_dir=directory)
    assert direct.to_dict() == config.to_dict()


def test_missing_image_is_a_configuration_error_before_device_initialization(document, tmp_path):
    document["devices"][0]["image_files"] = ["absent.png"]
    with pytest.raises(ConfigurationError, match="image_files"):
        CellConfig.from_dict(document, base_dir=tmp_path)


@pytest.mark.parametrize("text", ["{", "[]", '{"version":NaN,"devices":[]}'])
def test_file_loading_wraps_invalid_json_as_configuration_error(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_cell_config(path)


@pytest.mark.parametrize("factory,field", [
    (lambda: DeviceSpec("", "camera"), "function"),
    (lambda: DeviceSpec("CAM", "unknown"), "kind"),
    (lambda: DeviceSpec("CAM", "camera", {"seed": True}), "seed"),
    (lambda: DeviceSpec("PLC", "io", {"ports": True}), "ports"),
    (lambda: DeviceSpec("ARM", "duco", {"home_joints": [0] * 7}), "home_joints"),
    (lambda: CycleSpec(arrival_input=True), "arrival_input"),
    (lambda: CycleSpec(points=[{"pose": [0] * 5}]), "pose"),
    (lambda: CycleSpec(camera_timeout_ms=float("nan")), "camera_timeout_ms"),
    (lambda: CellConfig(devices=({"function": "X", "kind": "io"},), cycle=None), "devices"),
])
def test_direct_dataclass_construction_cannot_bypass_validation(factory, field):
    with pytest.raises(ConfigurationError) as caught:
        factory()
    assert field in str(caught.value)


def test_direct_cell_constructor_checks_references_and_port_bounds():
    devices = (DeviceSpec("IO_0", "io", {"ports": 1}), DeviceSpec("CAMERA_0", "camera"),
               DeviceSpec("FLEXIV_0", "flexiv"), DeviceSpec("WARNING_LIGHT_0", "warning_light"))
    with pytest.raises(ConfigurationError, match="arrival_input"):
        CellConfig(devices=devices, cycle=CycleSpec(arrival_input=8))
    with pytest.raises(ConfigurationError, match="camera"):
        CellConfig(devices=devices, cycle=CycleSpec(camera="MISSING"))


def test_defaults_do_not_share_mutable_options_or_waypoints():
    first, second = DeviceSpec("A", "camera"), DeviceSpec("B", "camera")
    assert first.options is not second.options
    one, two = CycleSpec(), CycleSpec()
    assert one.points is not two.points
    assert one.points[0] is not two.points[0]


@pytest.mark.parametrize("mode,source", [("Software", "Line0"), ("Hardware", "Software")])
def test_conflicting_explicit_trigger_mode_and_source_are_invalid(document, mode, source):
    document["devices"][0]["parameters"].update(TriggerMode=mode, TriggerSource=source)
    with pytest.raises(ConfigurationError, match="TriggerSource"):
        CellConfig.from_dict(document)


def cli_exit_code(main, arguments):
    try:
        return main(arguments)
    except SystemExit as error:
        return error.code


@pytest.mark.parametrize("command", ["validate", "serve", "demo"])
def test_invalid_cli_configuration_is_rejected_before_bind_or_thread_start(document, tmp_path, monkeypatch, command):
    import industrial_sim.__main__ as entry
    import industrial_sim.adapters.sensium as service
    import threading

    document["cycle"]["camera"] = "MISSING"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    operations = []

    def forbidden(*args, **kwargs):
        operations.append("resource-start")
        raise AssertionError("invalid configuration started a thread or bound a port")

    monkeypatch.setattr(service, "_Server", forbidden)
    monkeypatch.setattr(threading.Thread, "start", forbidden)
    args = [command, "--config", str(path)]
    if command == "demo":
        args += ["--output", str(tmp_path / "output")]
    try:
        code = cli_exit_code(entry.main, args)
    except ConfigurationError:
        code = 2
    assert code != 0
    assert operations == []


def test_cli_validate_is_read_only_and_config_robot_flags_conflict(document, tmp_path, monkeypatch):
    import industrial_sim.__main__ as entry
    import industrial_sim.adapters.sensium as service

    path = tmp_path / "cell.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("validation or conflicting flags bound a service port")

    monkeypatch.setattr(service, "_Server", forbidden)
    assert cli_exit_code(entry.main, ["validate", "--config", str(path)]) == 0
    code = cli_exit_code(entry.main, ["demo", "--config", str(path), "--robot", "duco",
                                      "--output", str(tmp_path / "output")])
    assert code != 0

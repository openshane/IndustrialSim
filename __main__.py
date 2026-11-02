"""Command line entry points for the local simulator and first closed loop."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import threading

import cv2

from industrial_sim.adapters.sensium import SimulatorServer
from industrial_sim.adapters.kuka_socket import KukaSocketSimulator
from industrial_sim.client import IndustrialSimClient
from industrial_sim.core.config import ConfigurationError, load_cell_config
from industrial_sim.scenarios.sensium_cell import run_cycle


def main(argv=None):
    parser = argparse.ArgumentParser(description="Programmable virtual devices for Sensium/GrindingTool")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="start the loopback device service")
    serve.add_argument("--port", type=int, default=18650)
    serve.add_argument("--config", type=Path, help="JSON device definitions; image paths are relative to this file")
    demo = commands.add_parser("demo", help="run arrival/camera/robot/I/O/light over HTTP")
    demo.add_argument("--url", help="use an existing local service instead of starting a temporary one")
    demo.add_argument("--robot", choices=("flexiv", "duco", "kuka"))
    demo.add_argument("--config", type=Path, help="JSON device and closed-loop definition")
    demo.add_argument("--fault", choices=("camera-timeout", "robot-fault", "arrival-timeout"))
    demo.add_argument("--output", type=Path, default=Path(".industrial-sim/last-run"))
    peer = commands.add_parser("kuka-peer", help="connect a virtual KUKA peer to GrindingTool's XML socket server")
    peer.add_argument("--port", type=int, required=True)
    peer.add_argument("--function", required=True, help="Function allowed by the project's socket_server configuration")
    peer.add_argument("--fields", default="{}", help="additional Robot request fields as a JSON object")
    peer.add_argument("--timeout", type=float, default=30)
    validate = commands.add_parser("validate", help="validate a cell without starting any devices or sockets")
    validate.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = None
    if getattr(args, "config", None) is not None:
        try:
            config = load_cell_config(args.config)
        except ConfigurationError as exc:
            parser.error(str(exc))
    if args.command == "validate":
        print(json.dumps({"valid": True, "devices": len(config.devices),
                          "cycle": config.cycle is not None, "config": config.to_dict()},
                         ensure_ascii=False, indent=2))
        return 0
    if args.command == "demo" and config is not None:
        if args.robot is not None:
            parser.error("--robot cannot be combined with --config; select the robot in cycle.robot")
        if config.cycle is None:
            parser.error("cycle: a closed-loop definition is required for demo")
    if args.command == "serve":
        stopping = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stopping.set())
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, lambda *_: stopping.set())
        with SimulatorServer(port=args.port, config=config) as server:
            print(json.dumps({"url": server.url, "protocol_version": 1}), flush=True)
            stopping.wait()
        return 0

    if args.command == "kuka-peer":
        fields = json.loads(args.fields)
        if not isinstance(fields, dict):
            parser.error("--fields must be a JSON object")
        result = KukaSocketSimulator(port=args.port, timeout=args.timeout).exchange(args.function, fields)
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0

    def execute(url):
        args.output.mkdir(parents=True, exist_ok=True)
        artifacts = []

        def save_image(image):
            # Python RGB8 frames use RGB channel order; image encoders expect
            # OpenCV BGR. Gray, Bayer and Mono16 arrays retain their raw layout.
            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            name = "capture.png" if not artifacts else f"capture-{len(artifacts) + 1:04d}.png"
            encoded_ok, encoded = cv2.imencode(".png", image)
            if not encoded_ok:
                raise RuntimeError("could not save captured image")
            encoded.tofile(str(args.output / name))
            artifacts.append(name)

        result = run_cycle(IndustrialSimClient(url), robot_kind=args.robot,
                           inject_arrival=args.fault != "arrival-timeout",
                           camera_timeout=args.fault == "camera-timeout",
                           robot_fault=args.fault == "robot-fault", image_sink=save_image, config=config)
        data = result.to_dict()
        data["artifacts"] = {"images": artifacts}
        data["configuration"] = None if config is None else config.to_dict()
        document = json.dumps(data, ensure_ascii=False, indent=2)
        (args.output / "result.json").write_text(document + "\n", encoding="utf-8")
        print(document)
        return 0 if result.success else 1

    if args.url:
        return execute(args.url)
    with SimulatorServer(config=config) as server:
        return execute(server.url)


if __name__ == "__main__":
    raise SystemExit(main())

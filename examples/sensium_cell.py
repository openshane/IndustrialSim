"""Construct a complete virtual cell from Python, without modifying GrindingTool.

Run after installing this repository: python examples/sensium_cell.py
"""

import json

from industrial_sim.adapters.sensium import SimulatorServer
from industrial_sim.client import IndustrialSimClient
from industrial_sim.core.config import CellConfig, CycleSpec, DeviceSpec
from industrial_sim.scenarios.sensium_cell import run_cycle


def main():
    config = CellConfig(
        devices=(
            DeviceSpec("arrival", "io", {"ports": 2}),
            DeviceSpec("inspection", "camera", {
                "seed": 42,
                "parameters": {"Width": 96, "Height": 64, "PixelFormat": "Mono8", "TriggerMode": "Software"},
            }),
            DeviceSpec("arm", "duco"),
            DeviceSpec("tower", "warning_light"),
        ),
        cycle=CycleSpec(io="arrival", camera="inspection", robot="arm", light="tower",
                        arrival_input=3, complete_output=4, running_output=5, fault_output=6),
    )

    def build_path(image):
        # Replace this with your vision/planning function; pose units are mm/deg.
        return [{"pose": [image.shape[1], image.shape[0], 500, 180, 0, 180]}]

    with SimulatorServer(config=config) as server:
        result = run_cycle(IndustrialSimClient(server.url), config=config, trajectory_builder=build_path)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())

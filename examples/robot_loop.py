"""Drive a robot with a served policy. Replace the two stubs with your robot's driver.

    vla-serve serve pi05-libero            # in another terminal (or: serve, then /v1/models/load)
    python examples/robot_loop.py --model pi05-libero --profile libero --task "put the bowl on the plate"
"""
import argparse

import numpy as np

from vlaserve.client import ChunkExecutor, PolicyClient
from vlaserve.profiles import RobotProfile
from vlaserve.spec import PolicySpec


def read_robot(profile: RobotProfile, spec: PolicySpec) -> dict:
    """STUB: return the robot's current camera frames (by the robot's names) and joint state."""
    frames = {robot_cam: np.zeros((480, 640, 3), np.uint8) for robot_cam in profile.cameras.values()}
    return {"images": profile.to_policy_images(frames),
            "state": np.zeros(spec.state_dim or 0, np.float32)}


def command_robot(action: np.ndarray) -> None:
    """STUB: send one absolute action (robot units, in the checkpoint's action_names order)."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11450")
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seconds", type=float, default=10)
    args = ap.parse_args()

    client = PolicyClient(args.url, model=args.model)
    client.load()                                   # no-op if already loaded
    spec_info = client.spec()
    spec = PolicySpec.from_dict(spec_info["spec"])
    profile = RobotProfile.load(args.profile)
    profile.require(spec)                           # refuse to move on a mismatch
    if spec_info["license_warning"]:
        print("licence:", spec_info["license_warning"])

    client.reset()                                  # new episode
    ex = ChunkExecutor(client, get_observation=lambda: read_robot(profile, spec),
                       send_action=command_robot, fps=profile.fps)
    stats = ex.run(task=args.task, seconds=args.seconds)
    print(stats.summary())


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Teleoperate one arm of the YAM Fold-Towel task with a physical SO-101 leader arm.

This script is meant for testing/benchmarking teleoperation latency: it reads joint
positions from a real SO-101 leader arm (via the ``lerobot`` package) every simulation
step and drives ``robot_1`` of the ``Isaac-Fold-Towel-Yam-Joint-v0`` task in real time.
``robot_2`` (the second YAM arm) is held at its initial resting pose the whole session.

The SO-101 leader has 5 arm joints + 1 gripper, while YAM has 6 arm joints + 1 gripper.
To bridge this DoF mismatch, one YAM joint is held constant (default: joint6, the last
wrist joint, fixed at 0.0 rad) and the SO-101's 5 remaining joints drive YAM's other 5
joints, in order (shoulder_pan -> joint1, shoulder_lift -> joint2, elbow_flex -> joint3,
wrist_flex -> joint4, wrist_roll -> joint5).

Requirements:
    Requires the ``lerobot`` package for SO-101 leader hardware access, in addition to
    the regular SoftMimicGen / Isaac Lab environment:

    .. code-block:: bash

        pip install lerobot      # or: uv pip install lerobot

Usage:

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/teleoperation/so101_leader_yam_teleop.py \
        --teleop_port /dev/ttyACM0 --teleop_id leader_arm_1 --enable_cameras

Note:
    ``--enable_cameras`` is required because the Fold-Towel task's scene defines camera
    sensors (top / left_wrist / right_wrist), which are included in the per-step timing.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Drive one arm of the YAM Fold-Towel task with a SO-101 leader arm.")
parser.add_argument("--task", type=str, default="Isaac-Fold-Towel-Yam-Joint-v0", help="Name of the task.")
parser.add_argument(
    "--teleop_port",
    type=str,
    default=os.getenv("TELEOP_PORT", "/dev/ttyACM0"),
    help="Serial port of the SO-101 leader arm.",
)
parser.add_argument(
    "--teleop_id",
    type=str,
    default=os.getenv("TELEOP_ID", "leader_arm_1"),
    help="Calibration id of the SO-101 leader arm.",
)
parser.add_argument(
    "--fixed_joint_index",
    type=int,
    default=5,
    help="0-based index (within joint1..joint6) of the YAM joint held constant. Default 5 = joint6.",
)
parser.add_argument(
    "--fixed_joint_value", type=float, default=0.0, help="Constant target (rad) for the fixed YAM joint."
)
parser.add_argument("--print_every", type=int, default=60, help="Print loop latency stats every N steps.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import gymnasium as gym
import torch

import softmimicgen_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

try:
    from lerobot.robots import make_robot_from_config
    from lerobot.teleoperators.so101_leader import SO101LeaderConfig
except ImportError as e:
    raise ImportError(
        "This script requires the 'lerobot' package for SO-101 leader hardware access. Install it with"
        " `pip install lerobot` (or `uv pip install lerobot`)."
    ) from e

# SO-101 leader keys, in the order they drive YAM's non-fixed arm joints.
SO101_LEADER_ARM_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
]
SO101_LEADER_GRIPPER_KEY = "gripper.pos"


class LatencyStats:
    """Tracks and periodically prints loop timing statistics."""

    def __init__(self, print_every: int):
        self.print_every = print_every
        self.count = 0
        self.last_print_time = time.perf_counter()
        self.read_time_total = 0.0
        self.step_time_total = 0.0

    def update(self, read_dt: float, step_dt: float) -> None:
        self.count += 1
        self.read_time_total += read_dt
        self.step_time_total += step_dt
        if self.count % self.print_every == 0:
            now = time.perf_counter()
            elapsed = now - self.last_print_time
            hz = self.print_every / elapsed if elapsed > 0 else 0.0
            avg_read_ms = 1000 * self.read_time_total / self.print_every
            avg_step_ms = 1000 * self.step_time_total / self.print_every
            avg_loop_ms = 1000 * elapsed / self.print_every
            print(
                f"[latency] loop={hz:6.1f} Hz | leader.get_action={avg_read_ms:6.2f} ms |"
                f" env.step={avg_step_ms:6.2f} ms | total/step={avg_loop_ms:6.2f} ms"
            )
            self.last_print_time = now
            self.read_time_total = 0.0
            self.step_time_total = 0.0


def main() -> None:
    # parse configuration (single env; terminations disabled so the session runs until Ctrl+C)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.terminations = None

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    device = env.device
    action_dim = env.action_manager.total_action_dim
    assert action_dim == 14, (
        f"Expected action dim 14 (arm_1[6]+gripper_1[1]+arm_2[6]+gripper_2[1]), got {action_dim}. The"
        " task's action layout may have changed - update the indexing in this script accordingly."
    )
    actions = torch.zeros(1, action_dim, device=device)
    # robot_2 (indices 7:13 = arm, 13 = gripper) stays at its initial resting pose for the whole
    # session: zero joint targets match its reset default_joint_pos, and the coupled gripper action
    # is held at 1.0 (open) to match the (0.04, -0.04) open finger pose set at reset. Unlike the arm
    # joints, the coupled gripper action is NOT offset-based - leaving it at 0.0 would immediately
    # command the gripper closed.
    actions[0, 13] = 1.0

    # resolve robot_1's controlled arm joints and their (safe) runtime joint limits
    robot_1 = env.scene["robot_1"]
    arm_joint_ids, arm_joint_names = robot_1.find_joints(["joint.*"])
    assert len(arm_joint_ids) == 6, f"Expected 6 arm joints on robot_1, found {arm_joint_names}"
    print(f"[INFO] robot_1 arm joint order: {arm_joint_names}")
    joint_limits = robot_1.data.soft_joint_pos_limits[0, arm_joint_ids, :].cpu()

    # non-fixed YAM joint indices, in the order the SO-101 leader's 5 arm joints drive them
    driven_joint_indices = [i for i in range(6) if i != args_cli.fixed_joint_index]
    assert len(driven_joint_indices) == len(SO101_LEADER_ARM_KEYS)

    # connect to the physical SO-101 leader arm
    leader_cfg = SO101LeaderConfig(port=args_cli.teleop_port, id=args_cli.teleop_id)
    leader = make_robot_from_config(leader_cfg)
    leader.connect()
    print(f"[INFO] Connected to SO-101 leader arm at {args_cli.teleop_port} (id={args_cli.teleop_id})")
    print("[INFO] Teleoperation running. Press Ctrl+C to stop.")

    stats = LatencyStats(args_cli.print_every)

    try:
        with torch.inference_mode():
            while simulation_app.is_running():
                t0 = time.perf_counter()
                leader_action = leader.get_action()
                t1 = time.perf_counter()

                for key, joint_idx in zip(SO101_LEADER_ARM_KEYS, driven_joint_indices):
                    raw = leader_action[key]
                    norm = min(max((raw + 100.0) / 200.0, 0.0), 1.0)
                    lo, hi = joint_limits[joint_idx]
                    actions[0, joint_idx] = lo + norm * (hi - lo)
                actions[0, args_cli.fixed_joint_index] = args_cli.fixed_joint_value

                gripper_raw = leader_action[SO101_LEADER_GRIPPER_KEY]
                actions[0, 6] = min(max(gripper_raw / 100.0, 0.0), 1.0)

                t2 = time.perf_counter()
                env.step(actions)
                t3 = time.perf_counter()

                stats.update(read_dt=t1 - t0, step_dt=t3 - t2)
    except KeyboardInterrupt:
        print("\n[INFO] Teleoperation interrupted by user.")
    finally:
        leader.disconnect()
        env.close()
        print("[INFO] Disconnected leader arm and closed environment.")


if __name__ == "__main__":
    main()
    simulation_app.close()

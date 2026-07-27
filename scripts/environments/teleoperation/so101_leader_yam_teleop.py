# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Teleoperate one arm of the YAM Fold-Towel task with a physical SO-101 leader arm.

This script is meant for testing/benchmarking teleoperation latency: it reads joint
positions from a real SO-101 leader arm every simulation step and drives ``robot_1``
of the ``Isaac-Fold-Towel-Yam-Joint-v0`` task in real time. ``robot_2`` (the second
YAM arm) is held at its initial resting pose the whole session.

The SO-101 leader has 5 arm joints + 1 gripper, while YAM has 6 arm joints + 1 gripper.
To bridge this DoF mismatch, one YAM joint is held constant (default: joint6, the last
wrist joint, fixed at 0.0 rad) and the SO-101's 5 remaining joints drive YAM's other 5
joints, in order (shoulder_pan -> joint1, shoulder_lift -> joint2, elbow_flex -> joint3,
wrist_flex -> joint4, wrist_roll -> joint5).

Two ``--leader_source`` modes are supported:

* ``local`` (default): connect directly to the SO-101 leader hardware via ``lerobot`` on
  the SAME machine that runs this script. Requires the ``lerobot`` package.
* ``network``: the SO-101 leader is attached to a DIFFERENT machine (e.g. this script
  runs on an EC2 GPU instance, while the leader is plugged into your local Mac). Run
  ``so101_leader_client.py`` on the machine with the leader attached; it streams leader
  joint positions over TCP to this script, which listens on ``--listen_host``/``--listen_port``
  and does NOT need ``lerobot`` installed at all.

  TCP (rather than UDP) is used deliberately so the stream can be carried over an ordinary
  SSH local port-forward (``ssh -L``), which only forwards TCP. ``TCP_NODELAY`` is set on
  both ends to disable Nagle's algorithm - without it, small packets can be buffered for up
  to ~40 ms, which would dominate the very latency this script is meant to measure.

Requirements (``local`` mode only):

    .. code-block:: bash

        pip install lerobot      # or: uv pip install lerobot

Usage (local mode, leader hardware attached to this machine):

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/teleoperation/so101_leader_yam_teleop.py \
        --leader_source local --teleop_port /dev/ttyACM0 --teleop_id leader_arm_1 --enable_cameras

Usage (network mode, e.g. this script on EC2, leader on your local Mac):

.. code-block:: bash

    # on EC2 (this script) - open an SSH tunnel from your Mac first, or open the
    # security group port, then:
    ./isaaclab.sh -p scripts/environments/teleoperation/so101_leader_yam_teleop.py \
        --leader_source network --listen_host 0.0.0.0 --listen_port 9999 --enable_cameras

    # on your Mac (so101_leader_client.py, no Isaac Sim required):
    #   ssh -L 9999:localhost:9999 <user>@<ec2-host>   # in a separate terminal, keep open
    python scripts/environments/teleoperation/so101_leader_client.py \
        --host localhost --port 9999 --teleop_port /dev/tty.usbmodemXXXX --teleop_id leader_arm_1
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
    "--leader_source",
    type=str,
    default="local",
    choices=["local", "network"],
    help=(
        "Where SO-101 leader joint positions come from. 'local': connect directly to the leader"
        " hardware via lerobot on this machine (requires --teleop_port/--teleop_id and 'lerobot'"
        " installed). 'network': receive leader packets over UDP from so101_leader_client.py"
        " running on a different machine (e.g. this script on an EC2 instance, the client on your"
        " Mac with the leader attached) - no 'lerobot' needed here."
    ),
)
parser.add_argument(
    "--teleop_port",
    type=str,
    default=os.getenv("TELEOP_PORT", "/dev/ttyACM0"),
    help="[local mode] Serial port of the SO-101 leader arm.",
)
parser.add_argument(
    "--teleop_id",
    type=str,
    default=os.getenv("TELEOP_ID", "leader_arm_1"),
    help="[local mode] Calibration id of the SO-101 leader arm.",
)
parser.add_argument(
    "--listen_host",
    type=str,
    default="127.0.0.1",
    help=(
        "[network mode] Host/interface to listen for leader packets on. Default 127.0.0.1 is"
        " correct when the client reaches this machine through an SSH port-forward. Use 0.0.0.0"
        " only if the client connects directly to this machine's public address."
    ),
)
parser.add_argument(
    "--listen_port",
    type=int,
    default=9999,
    help="[network mode] TCP port to listen for leader packets on.",
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

import json
import socket
import threading
import time

import gymnasium as gym
import torch

import softmimicgen_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

if args_cli.leader_source == "local":
    try:
        from lerobot.robots import make_robot_from_config
        from lerobot.teleoperators.so101_leader import SO101LeaderConfig
    except ImportError as e:
        raise ImportError(
            "This script requires the 'lerobot' package for SO-101 leader hardware access. Install it with"
            " `pip install lerobot` (or `uv pip install lerobot`), or use --leader_source network instead."
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


class NetworkLeaderReceiver:
    """Background TCP receiver that keeps the most recently received SO-101 leader packet.

    Accepts a connection from ``so101_leader_client.py`` and runs a daemon thread that
    continuously reads newline-delimited JSON leader action dicts, storing only the most
    recent one. The main simulation loop then reads the latest value every step without
    blocking on network I/O, so network jitter shows up as "packet age" rather than
    stalling the sim loop.

    TCP is used (rather than UDP) so the stream can be tunnelled through an ordinary SSH
    local port-forward, which only carries TCP. ``TCP_NODELAY`` disables Nagle's algorithm
    so that each small joint-position packet is sent immediately instead of being buffered.

    If the client disconnects, the receiver goes back to waiting for a new connection, so
    the sender can be restarted without restarting the simulation.
    """

    def __init__(self, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(1)
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._latest_recv_time: float | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._sock.close()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                conn, addr = self._sock.accept()
            except OSError:
                break
            print(f"[INFO] Leader client connected from {addr[0]}:{addr[1]}")
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with conn:
                self._read_connection(conn)
            print("[INFO] Leader client disconnected. Waiting for a new connection...")

    def _read_connection(self, conn: socket.socket) -> None:
        buffer = b""
        while not self._stop_event.is_set():
            try:
                chunk = conn.recv(4096)
            except OSError:
                return
            if not chunk:
                return
            buffer += chunk
            # the sender delimits each JSON payload with a newline; a single recv() may
            # contain several payloads or only part of one, so parse line by line.
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    action = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                with self._lock:
                    self._latest = action
                    self._latest_recv_time = time.perf_counter()

    def get_latest(self) -> tuple[dict | None, float | None]:
        """Returns (latest_action, seconds_since_received), or (None, None) if nothing received yet."""
        with self._lock:
            if self._latest is None:
                return None, None
            return self._latest, time.perf_counter() - self._latest_recv_time


class LatencyStats:
    """Tracks and periodically prints loop timing statistics."""

    def __init__(self, print_every: int, read_label: str = "leader read"):
        self.print_every = print_every
        self.read_label = read_label
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
                f"[latency] loop={hz:6.1f} Hz | {self.read_label}={avg_read_ms:6.2f} ms |"
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

    # set up the leader source: either connect directly to local SO-101 hardware, or listen
    # for packets streamed over the network by so101_leader_client.py
    leader = None
    receiver = None
    if args_cli.leader_source == "local":
        leader_cfg = SO101LeaderConfig(port=args_cli.teleop_port, id=args_cli.teleop_id)
        leader = make_robot_from_config(leader_cfg)
        leader.connect()
        print(f"[INFO] Connected to SO-101 leader arm at {args_cli.teleop_port} (id={args_cli.teleop_id})")

        def read_leader_action() -> tuple[dict, float]:
            t0 = time.perf_counter()
            action = leader.get_action()
            return action, time.perf_counter() - t0

        read_label = "leader.get_action"
    else:
        receiver = NetworkLeaderReceiver(args_cli.listen_host, args_cli.listen_port)
        receiver.start()
        print(f"[INFO] Listening for SO-101 leader packets on tcp://{args_cli.listen_host}:{args_cli.listen_port}")
        print(
            "[INFO] Waiting for the first packet (run so101_leader_client.py on the machine with the"
            " SO-101 leader attached)..."
        )
        while True:
            action, age = receiver.get_latest()
            if action is not None:
                break
            time.sleep(0.05)
        print("[INFO] First packet received.")

        def read_leader_action() -> tuple[dict, float]:
            action, age = receiver.get_latest()
            return action, age if age is not None else 0.0

        read_label = "packet age"

    print("[INFO] Teleoperation running. Press Ctrl+C to stop.")

    stats = LatencyStats(args_cli.print_every, read_label)

    try:
        with torch.inference_mode():
            while simulation_app.is_running():
                leader_action, read_dt = read_leader_action()

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

                stats.update(read_dt=read_dt, step_dt=t3 - t2)
    except KeyboardInterrupt:
        print("\n[INFO] Teleoperation interrupted by user.")
    finally:
        if leader is not None:
            leader.disconnect()
        if receiver is not None:
            receiver.stop()
        env.close()
        print("[INFO] Disconnected leader arm and closed environment.")


if __name__ == "__main__":
    main()
    simulation_app.close()

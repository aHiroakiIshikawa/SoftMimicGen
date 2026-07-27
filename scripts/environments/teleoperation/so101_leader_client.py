# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stream a physical SO-101 leader arm's joint positions over TCP to a remote
``so101_leader_yam_teleop.py --leader_source network`` process (e.g. running on an
EC2 GPU instance while the SO-101 leader is plugged into this machine).

This script has NO Isaac Sim / Isaac Lab dependency - it only needs the ``lerobot``
package and the Python standard library, so it can run in a lightweight local
environment (e.g. a plain uv/venv on your Mac) without installing Isaac Sim locally.

Each reading is sent as one newline-delimited JSON object. TCP is used (rather than UDP)
so the stream can be carried over an ordinary SSH local port-forward (``ssh -L``), which
only forwards TCP. ``TCP_NODELAY`` is set to disable Nagle's algorithm, otherwise these
small packets could be buffered for up to ~40 ms and distort the latency measurement.

Requirements:

    .. code-block:: bash

        pip install lerobot      # or: uv pip install lerobot

Usage (via an SSH tunnel - recommended, no EC2 security group port needs to be opened):

.. code-block:: bash

    # in a separate terminal, keep this open for the duration of the session:
    ssh -L 9999:localhost:9999 <user>@<ec2-host>

    # then point the client at the local end of the tunnel:
    python so101_leader_client.py --host localhost --port 9999 \
        --teleop_port /dev/tty.usbmodemXXXX --teleop_id leader_arm_1

Usage (direct connection, remote host reachable on the network):

.. code-block:: bash

    python so101_leader_client.py --host <EC2_HOST_OR_IP> --port 9999 \
        --teleop_port /dev/tty.usbmodemXXXX --teleop_id leader_arm_1
"""

import argparse
import json
import os
import socket
import time

parser = argparse.ArgumentParser(description="Stream SO-101 leader arm joint positions over TCP.")
parser.add_argument(
    "--host",
    type=str,
    required=True,
    help="Hostname/IP of the receiving so101_leader_yam_teleop.py process (or 'localhost' if using an SSH tunnel).",
)
parser.add_argument("--port", type=int, default=9999, help="TCP port of the receiving process.")
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
    "--hz", type=float, default=0.0, help="Target send rate in Hz. 0 (default) = as fast as possible, no sleep."
)
parser.add_argument("--print_every", type=int, default=60, help="Print send-rate stats every N packets.")
args_cli = parser.parse_args()

try:
    from lerobot.robots import make_robot_from_config

    try:
        # lerobot >= 0.5.2 merged the SO-100/101 leaders into a single ``so_leader`` module.
        from lerobot.teleoperators.so_leader import SO101LeaderConfig
    except ImportError:
        # older lerobot releases kept a dedicated ``so101_leader`` module.
        from lerobot.teleoperators.so101_leader import SO101LeaderConfig
except ImportError as e:
    raise ImportError(
        "This script requires the 'lerobot' package for SO-101 leader hardware access. Install it with"
        " `pip install lerobot` (or `uv pip install lerobot`)."
    ) from e


def main() -> None:
    # use_degrees=False -> arm joints are reported in lerobot's normalized -100..100 range
    # (the gripper stays 0..100). The receiving side maps that range onto the YAM joint limits,
    # so it must NOT be left at the SO-101 default of use_degrees=True.
    leader_cfg = SO101LeaderConfig(port=args_cli.teleop_port, id=args_cli.teleop_id, use_degrees=False)
    leader = make_robot_from_config(leader_cfg)
    leader.connect()
    print(f"[INFO] Connected to SO-101 leader arm at {args_cli.teleop_port} (id={args_cli.teleop_id})")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # disable Nagle's algorithm: these packets are tiny and must go out immediately,
    # otherwise the kernel may hold them back and add tens of ms of latency.
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[INFO] Connecting to tcp://{args_cli.host}:{args_cli.port} ...")
    sock.connect((args_cli.host, args_cli.port))
    print("[INFO] Connected. Streaming leader joint positions.")
    print("[INFO] Press Ctrl+C to stop.")

    period = 1.0 / args_cli.hz if args_cli.hz > 0 else 0.0
    count = 0
    last_print_time = time.perf_counter()

    try:
        while True:
            t0 = time.perf_counter()
            leader_action = leader.get_action()
            # add a wall-clock send timestamp so the receiver can (approximately, modulo clock
            # sync between the two machines) estimate one-way network + read latency.
            payload = dict(leader_action)
            payload["_t"] = time.time()
            # newline-delimited JSON: TCP is a byte stream with no message boundaries, so the
            # receiver needs an explicit delimiter to split consecutive readings.
            sock.sendall(json.dumps(payload).encode("utf-8") + b"\n")

            count += 1
            if count % args_cli.print_every == 0:
                now = time.perf_counter()
                elapsed = now - last_print_time
                hz = args_cli.print_every / elapsed if elapsed > 0 else 0.0
                print(f"[send] {hz:6.1f} Hz")
                last_print_time = now

            if period > 0:
                elapsed = time.perf_counter() - t0
                if elapsed < period:
                    time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except (BrokenPipeError, ConnectionResetError):
        print("\n[ERROR] Connection to the receiver was lost.")
    finally:
        leader.disconnect()
        sock.close()
        print("[INFO] Disconnected leader arm.")


if __name__ == "__main__":
    main()

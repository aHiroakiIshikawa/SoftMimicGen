# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, List

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_multiple_assets_root_state_uniform_same_ranges(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: List[SceneEntityCfg],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    randomize_jenga_z: bool = False,
):
    """Reset multiple assets' root states using the same randomization ranges and values.

    Args:
        env: The environment.
        env_ids: Environment IDs to reset.
        asset_cfgs: List of asset configurations to randomize.
        pose_range: Pose randomization ranges per axis.
        velocity_range: Velocity randomization ranges per axis.
        randomize_jenga_z: Whether to randomize the jenga block Z between layer heights.
    """
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    pose_rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    velocity_rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    for asset_cfg in asset_cfgs:
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        root_states = asset.data.default_root_state[env_ids].clone()

        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pose_rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(pose_rand_samples[:, 3], pose_rand_samples[:, 4], pose_rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

        velocities = root_states[:, 7:13] + velocity_rand_samples

        if asset_cfg.name == "jenga" and randomize_jenga_z:
            jenga_z_values = torch.tensor([0.0854, 0.2002], device=env.device)
            z_indices = torch.randint(0, 2, (len(env_ids),), device=env.device)
            positions[:, 2] = jenga_z_values[z_indices]

        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

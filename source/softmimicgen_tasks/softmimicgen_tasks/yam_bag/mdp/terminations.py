# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, Deformable, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_in_bag(
    env: ManagerBasedRLEnv,
    robot_1_cfg: SceneEntityCfg = SceneEntityCfg("robot_1"),
    robot_2_cfg: SceneEntityCfg = SceneEntityCfg("robot_2"),
    banana_cfg: SceneEntityCfg = SceneEntityCfg("banana"),
    bag_cfg: SceneEntityCfg = SceneEntityCfg("bag"),
    containment_radius: float = 0.22,
    velocity_threshold: float = 0.2,
    gripper_open_val: float = 0.04,
    gripper_tolerance: float = 0.01,
) -> torch.Tensor:
    """Check if rigid object is inside the deformable bag.

    Uses the distance from object origin to the bag's nodal centroid.

    Args:
        env: The environment.
        robot_1_cfg: First robot configuration.
        robot_2_cfg: Second robot configuration.
        banana_cfg: Rigid object configuration.
        bag_cfg: Deformable bag configuration.
        containment_radius: Max distance from object to bag centroid.
        velocity_threshold: Max object speed to count as settled.
        gripper_open_val: Joint position when gripper is fully open.
        gripper_tolerance: Tolerance for gripper open check.
    """
    banana: RigidObject = env.scene[banana_cfg.name]
    bag: Deformable = env.scene[bag_cfg.name]
    robot_1: Articulation = env.scene[robot_1_cfg.name]
    robot_2: Articulation = env.scene[robot_2_cfg.name]

    banana_pos = banana.data.root_pos_w
    banana_vel = banana.data.root_lin_vel_w
    bag_centroid = bag.data.nodal_pos_w.mean(dim=1)

    dist_to_centroid = torch.norm(banana_pos - bag_centroid, dim=-1)
    banana_speed = torch.norm(banana_vel, dim=-1)

    gripper_1_open = torch.isclose(
        robot_1.data.joint_pos[:, -2],
        torch.tensor(gripper_open_val, device=env.device),
        atol=gripper_tolerance, rtol=gripper_tolerance,
    )
    gripper_2_open = torch.isclose(
        robot_2.data.joint_pos[:, -2],
        torch.tensor(gripper_open_val, device=env.device),
        atol=gripper_tolerance, rtol=gripper_tolerance,
    )

    return (
        (dist_to_centroid < containment_radius)
        & (banana_speed < velocity_threshold)
        & gripper_1_open
        & gripper_2_open
    )

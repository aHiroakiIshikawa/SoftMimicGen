# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Deformable, DeformableObject, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

L_HAND_JOINT_NAMES = [
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_distal_joint",
]


def object_reached_goal(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
    xy_threshold: float = 0.1,
    height_threshold: float = 0.1,
    min_vel: float = 0.20,
    left_wrist_max_x: float = 0.26,
    finger_joint_threshold: float = -0.4,
    left_wrist_min_z: float = 1.15,
) -> torch.Tensor:
    """Check if the object has been placed on the container and the hand has retracted.

    Args:
        env: The environment.
        object_cfg: Deformable object configuration.
        container_cfg: Rigid container configuration.
        xy_threshold: Max xy distance between object and container.
        height_threshold: Max height difference between object and container.
        min_vel: Max object velocity per axis.
        left_wrist_max_x: Max wrist x-position (hand retracted laterally).
        finger_joint_threshold: Joint value above which a finger is considered open.
        left_wrist_min_z: Min wrist z-position (hand retracted vertically).
    """
    object: DeformableObject = env.scene[object_cfg.name]
    container: RigidObject = env.scene[container_cfg.name]

    pos_diff = object.data.root_pos_w - container.data.root_pos_w
    object_vel = torch.abs(object.data.root_vel_w)

    robot_body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_hand_roll_link")
    left_wrist_x = robot_body_pos_w[:, left_eef_idx, 0] - env.scene.env_origins[:, 0]
    left_wrist_z = robot_body_pos_w[:, left_eef_idx, 2] - env.scene.env_origins[:, 2]

    robot_joint_names = env.scene["robot"].data.joint_names
    indexes = torch.tensor([robot_joint_names.index(n) for n in L_HAND_JOINT_NAMES], dtype=torch.long)
    hand_joint_states = env.scene["robot"].data.joint_pos[:, indexes]

    xy_dist = torch.norm(pos_diff[:, :2], dim=1)
    h_dist = torch.norm(pos_diff[:, 2:], dim=1)

    joints_open = hand_joint_states > finger_joint_threshold
    finger_open = torch.sum(joints_open, dim=1) >= (hand_joint_states.shape[1] // 2)

    return (
        (xy_dist < xy_threshold)
        & (h_dist < height_threshold)
        & (object_vel[:, 0] < min_vel)
        & (object_vel[:, 1] < min_vel)
        & (object_vel[:, 2] < min_vel)
        & (left_wrist_x < left_wrist_max_x)
        & (left_wrist_z > left_wrist_min_z)
        & finger_open
    )


def object_flat(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    height_threshold: float = 1.035,
    wrist_min_z: float = 1.1,
) -> torch.Tensor:
    """Check if the towel is flat on the table and both hands have retracted.

    Args:
        env: The environment.
        object_cfg: Deformable object configuration.
        height_threshold: Max z-position for all nodes (towel lies flat).
        wrist_min_z: Min wrist z-position (hands retracted vertically).
    """
    object: Deformable = env.scene[object_cfg.name]
    nodal_pos_w = object.data.nodal_pos_w

    robot_body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_hand_roll_link")
    right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_roll_link")

    left_wrist_z = robot_body_pos_w[:, left_eef_idx, 2] - env.scene.env_origins[:, 2]
    right_wrist_z = robot_body_pos_w[:, right_eef_idx, 2] - env.scene.env_origins[:, 2]

    towel_flat = (nodal_pos_w[..., 2] <= height_threshold).all(dim=-1)

    return (left_wrist_z > wrist_min_z) & (right_wrist_z > wrist_min_z) & towel_flat

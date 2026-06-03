# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import DeformableObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .terminations import L_HAND_JOINT_NAMES


def object_grasped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.05,
    finger_joint_threshold: float = -0.4,
) -> torch.Tensor:
    """Check if the deformable object is grasped by the left hand.

    Grasp is detected when the closest node is within ``diff_threshold`` and
    at least half of the finger joints are closed.

    Args:
        env: The environment.
        object_cfg: Deformable object configuration.
        diff_threshold: Max distance between hand and closest node for grasp.
        finger_joint_threshold: Joint value at or below which a finger is considered closed.
    """
    object: DeformableObject = env.scene[object_cfg.name]

    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index("L_middle_proximal_link")
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins

    robot_joint_names = env.scene["robot"].data.joint_names
    indexes = torch.tensor([robot_joint_names.index(n) for n in L_HAND_JOINT_NAMES], dtype=torch.long)
    hand_joint_states = env.scene["robot"].data.joint_pos[:, indexes]

    nodal_pos_w = object.data.nodal_pos_w
    pose_diff = torch.norm(nodal_pos_w - left_eef_pos.unsqueeze(1), dim=-1)
    min_distance = pose_diff.min(dim=1).values

    joints_closed = hand_joint_states <= finger_joint_threshold
    finger_closed = torch.sum(joints_closed, dim=1) >= (hand_joint_states.shape[1] // 2)

    return (min_distance < diff_threshold) & finger_closed

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import DeformableObject, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ring_cfg: SceneEntityCfg = SceneEntityCfg("ring"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    x_threshold: float = 0.005,
    ring_radius: float = 0.01,
) -> torch.Tensor:
    """Check if the thread has been passed through the ring.

    Args:
        env: The environment.
        object_cfg: Deformable thread configuration.
        ring_cfg: Rigid ring configuration.
        ee_frame_cfg: End effector frame configuration.
        x_threshold: Min x-position in ring frame to count as passed through.
        ring_radius: Ring opening radius for y/z bounds.
    """
    object: DeformableObject = env.scene[object_cfg.name]
    ring: RigidObject = env.scene[ring_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    object_nodal_pos_w = object.data.nodal_pos_w
    ring_pos_w = ring.data.root_pos_w
    ring_quat_w = ring.data.root_quat_w

    object_nodal_pos_b, _ = subtract_frame_transforms(ring_pos_w, ring_quat_w, object_nodal_pos_w)
    ee_frame_pos_b, _ = subtract_frame_transforms(ring_pos_w, ring_quat_w, ee_frame_pos)

    x_condition = object_nodal_pos_b[..., 0] >= x_threshold
    y_condition = torch.abs(object_nodal_pos_b[..., 1]) <= ring_radius
    z_condition = torch.abs(object_nodal_pos_b[..., 2]) <= ring_radius
    ee_not_through_ring = ee_frame_pos_b[..., 0] < 0.0

    goal_condition = x_condition & y_condition & z_condition & ee_not_through_ring

    return torch.any(goal_condition, dim=1)

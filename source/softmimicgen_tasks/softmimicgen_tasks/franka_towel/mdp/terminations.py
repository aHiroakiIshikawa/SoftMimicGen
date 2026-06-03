# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, Deformable
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_folded(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    fold_threshold: float = 0.55,
    height_threshold: float = 0.12,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    atol=0.0001,
    rtol=0.0001,
) -> torch.Tensor:
    """Fold detection using PCA-based principal extent reduction.

    Computes the largest principal axis extent of the object's XY projection
    (rotation-invariant) and compares to the undeformed extent. Success requires
    the fold ratio to be below threshold, gripper to be open, and end effector
    to be retracted above the table.

    Args:
        env: The environment.
        robot_cfg: Robot configuration.
        object_cfg: Deformable object configuration.
        ee_frame_cfg: End effector frame configuration.
        fold_threshold: Ratio of current/initial extent for success.
        height_threshold: Minimum EE height for hand retraction check.
        gripper_open_val: Joint position when gripper is fully open.
        atol: Absolute tolerance for gripper position check.
        rtol: Relative tolerance for gripper position check.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: Deformable = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    nodal_pos_w = object.data.nodal_pos_w

    xy_pos = nodal_pos_w[..., :2]
    centered = xy_pos - xy_pos.mean(dim=1, keepdim=True)
    cov = torch.bmm(centered.transpose(1, 2), centered) / centered.shape[1]
    current_extent = torch.sqrt(torch.linalg.eigvalsh(cov)[:, -1])

    default_xy = object.data.default_nodal_state_w[..., :2]
    default_centered = default_xy - default_xy.mean(dim=1, keepdim=True)
    default_cov = torch.bmm(default_centered.transpose(1, 2), default_centered) / default_centered.shape[1]
    initial_extent = torch.sqrt(torch.linalg.eigvalsh(default_cov)[:, -1])

    fold_ratio = current_extent / initial_extent.clamp(min=1e-6)
    is_folded = fold_ratio < fold_threshold

    gripper_open = torch.isclose(
        robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=atol, rtol=rtol
    ) & torch.isclose(
        robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=atol, rtol=rtol
    )

    ee_retracted = ee_frame_pos[..., 2] >= height_threshold

    return is_folded & gripper_open & ee_retracted

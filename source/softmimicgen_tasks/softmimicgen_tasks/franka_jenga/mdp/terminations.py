# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.2,
    object_cfg: SceneEntityCfg = SceneEntityCfg("jenga"),
    jenga_block_cfg: SceneEntityCfg = SceneEntityCfg("jenga_block"),
) -> torch.Tensor:
    """Termination based on XY distance between object and reference block.

    Args:
        env: The environment.
        distance_threshold: Minimum XY separation between object and reference block required for success.
        object_cfg: The object to track.
        jenga_block_cfg: The reference block configuration.
    """
    block: RigidObject = env.scene[object_cfg.name]
    jenga_block: RigidObject = env.scene[jenga_block_cfg.name]

    block_pos = block.data.root_pos_w[..., :2]
    jenga_block_pos = jenga_block.data.root_pos_w[..., :2]

    xy_distance = torch.norm(block_pos - jenga_block_pos, dim=-1)

    return xy_distance >= distance_threshold

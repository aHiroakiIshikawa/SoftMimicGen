# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import DeformableObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.035,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check if any node of the deformable object has been lifted above the height threshold.

    Args:
        env: The environment.
        height_threshold: Minimum z-position for a node to count as lifted.
        object_cfg: The deformable object configuration.
    """
    object: DeformableObject = env.scene[object_cfg.name]
    nodal_positions = object.data.nodal_pos_w

    return torch.any(nodal_positions[..., 2] >= height_threshold, dim=1)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import DeformableObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_nodal_kinematic_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset kinematic targets for the tissue's corner anchor nodes.

    Args:
        env: The environment.
        env_ids: Environment indices to reset.
        asset_cfg: The deformable object configuration.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]

    default_nodal_state = asset.data.nodal_pos_w[env_ids].clone()
    nodal_kinematic_target = asset.data.nodal_kinematic_target[env_ids].clone()

    # Corner anchor node indices (mesh-specific)
    indices = [25, 27, 528, 529]
    nodal_kinematic_target[..., :3] = default_nodal_state[..., :3]
    nodal_kinematic_target[..., indices, 3] = 0.0

    asset.write_nodal_kinematic_target_to_sim(nodal_kinematic_target, env_ids=env_ids)

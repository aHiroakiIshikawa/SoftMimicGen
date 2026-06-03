# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import DeformableObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rope_ends_close_tracked(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    distance_threshold: float = 0.08,
    distance_threshold_min: float = 0.02,
    num_end_nodes: int = 3,
):
    """Check if the two ends of a rope are close to each other.

    Identifies rope ends at the start of each episode by finding the two
    farthest-apart nodes, then tracks those indices throughout the episode.

    Args:
        env: The environment.
        object_cfg: Configuration for the rope object.
        distance_threshold: Maximum distance between rope ends for success.
        distance_threshold_min: Minimum distance to avoid false positives.
        num_end_nodes: Number of nodes to consider at each end for robustness.

    Returns:
        Boolean tensor of shape (num_envs,) indicating if rope ends are close.
    """
    rope: DeformableObject = env.scene[object_cfg.name]

    nodal_positions = rope.data.nodal_pos_w
    num_envs, num_nodes, _ = nodal_positions.shape

    rope_ends_close = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    if not hasattr(env, '_rope_end_indices'):
        env._rope_end_indices = {}

    for env_idx in range(num_envs):
        env_nodal_pos = nodal_positions[env_idx]

        if env_idx not in env._rope_end_indices:
            pos_i = env_nodal_pos.unsqueeze(1)
            pos_j = env_nodal_pos.unsqueeze(0)
            distances = torch.norm(pos_i - pos_j, dim=2)

            max_dist_idx = torch.argmax(distances)
            end1_idx = max_dist_idx // num_nodes
            end2_idx = max_dist_idx % num_nodes

            env._rope_end_indices[env_idx] = {
                'end1_idx': end1_idx.item(),
                'end2_idx': end2_idx.item()
            }

        end1_idx = env._rope_end_indices[env_idx]['end1_idx']
        end2_idx = env._rope_end_indices[env_idx]['end2_idx']

        end1_pos = env_nodal_pos[end1_idx]
        end2_pos = env_nodal_pos[end2_idx]

        dist_to_end1 = torch.norm(env_nodal_pos - end1_pos, dim=1)
        dist_to_end2 = torch.norm(env_nodal_pos - end2_pos, dim=1)

        _, end1_node_indices = torch.topk(dist_to_end1, num_end_nodes, largest=False)
        _, end2_node_indices = torch.topk(dist_to_end2, num_end_nodes, largest=False)

        end1_nodes_pos = env_nodal_pos[end1_node_indices]
        end2_nodes_pos = env_nodal_pos[end2_node_indices]

        end1_expanded = end1_nodes_pos.unsqueeze(1)
        end2_expanded = end2_nodes_pos.unsqueeze(0)
        pairwise_distances = torch.norm(end1_expanded - end2_expanded, dim=2)

        min_end_distance = torch.min(pairwise_distances)

        rope_ends_close[env_idx] = (min_end_distance > distance_threshold_min) & (min_end_distance < distance_threshold)

    return rope_ends_close

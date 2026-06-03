# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import DeformableObject
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def reset_rope_end_tracking(env: ManagerBasedRLEnv, env_ids=None):
    """Reset the rope end tracking when episodes reset.

    Args:
        env: The environment object.
        env_ids: Environment IDs to reset.
    """
    if hasattr(env, '_rope_end_indices'):
        if env_ids is not None:
            for env_id in env_ids:
                if env_id in env._rope_end_indices:
                    del env._rope_end_indices[env_id]
        else:
            env._rope_end_indices.clear()


def reset_nodal_pose_portion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    partial_pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset asset nodal state with additional partial rotation on a subset of nodes.

    Applies a global 6-DOF transform to all nodes, then applies a second independent
    transform to a specified subset of nodes (e.g., one half of a rope) around their
    own centroid.

    Args:
        env: The environment.
        env_ids: Environment IDs to reset.
        pose_range: Global 6-DOF pose range for all nodes.
        partial_pose_range: Additional 6-DOF pose range applied to the node subset.
        asset_cfg: The deformable asset configuration.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    nodal_state = asset.data.default_nodal_state_w[env_ids].clone()

    # Global transform for all nodes
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    position = rand_samples[:, 0:3]
    quaternion = math_utils.quat_from_euler_xyz(
        roll=rand_samples[:, 3],
        pitch=rand_samples[:, 4],
        yaw=rand_samples[:, 5]
    )

    nodal_state[..., :3] = asset.transform_nodal_pos(nodal_state[..., :3], position, quaternion)

    # Additional transform for node subset (one half of the rope)
    range_list = [partial_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    position = rand_samples[:, 0:3]
    quaternion = math_utils.quat_from_euler_xyz(
        roll=rand_samples[:, 3],
        pitch=rand_samples[:, 4],
        yaw=rand_samples[:, 5]
    )

    # Node indices for one half of the rope (mesh-resolution specific)
    indices = [222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 270, 271,
        272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
        286, 287, 288, 289, 290, 291, 296, 297, 298, 299, 300, 301, 302, 303,
        304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317,
        318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 331, 332,
        333, 334, 335, 336, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347,
        348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361,
        362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375,
        376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389,
        390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403,
        404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
        418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431,
        432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445,
        446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,
        460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473,
        474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487,
        488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501,
        502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515,
        516, 517, 518, 519, 520, 528, 529, 530, 531, 532, 533, 534, 535, 536,
        537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548]

    subset_nodal_state = nodal_state[..., indices, :3]
    transformed_subset = asset.transform_nodal_pos(subset_nodal_state, position, quaternion)
    nodal_state[..., indices, :3] = transformed_subset

    asset.write_nodal_pos_to_sim(nodal_state[..., :3], env_ids=env_ids)

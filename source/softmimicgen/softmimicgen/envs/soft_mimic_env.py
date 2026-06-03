# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extended mimic environment with deformable object support."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class SoftManagerBasedRLMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic environment extended with deformable object support."""

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments and optionally run settling steps for deformable objects."""
        super()._reset_idx(env_ids)

        if getattr(self.cfg, "settling_steps", 0) > 0:
            zero_actions = torch.zeros(
                (self.num_envs, self.action_manager.total_action_dim), device=self.device
            )
            for _ in range(getattr(self.cfg, "settling_steps", 0)):
                self.action_manager.process_action(zero_actions)
                for _ in range(self.cfg.decimation):
                    self.action_manager.apply_action()
                    self.scene.write_data_to_sim()
                    self.sim.step(render=True)
                    self.scene.update(dt=self.physics_dt)

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """Gets the pose of each object relevant to Isaac Lab Mimic data generation in the current scene.

        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A dictionary that maps object names to object pose matrix (4x4 torch.Tensor)
        """
        if env_ids is None:
            env_ids = slice(None)

        object_pose_matrix = dict()

        rigid_object_states = self.scene.get_state(is_relative=True)["rigid_object"]

        if rigid_object_states:
            for obj_name, obj_state in rigid_object_states.items():
                object_pose_matrix[obj_name] = PoseUtils.make_pose(
                    obj_state["root_pose"][env_ids, :3],
                    PoseUtils.matrix_from_quat(obj_state["root_pose"][env_ids, 3:7]),
                )

        return object_pose_matrix

    def get_object_nodal_positions(self, env_ids: Sequence[int] | None = None):
        """Gets the nodal positions of each deformable object relevant to Isaac Lab Mimic data generation in the current scene.

        Args:
            env_ids: Environment indices to get the nodal positions for. If None, all envs are considered.

        Returns:
            A dictionary that maps object names to nodal position tensors.
            Shape is (len(env_ids), num_nodal_points, 3) for each object.
        """
        if env_ids is None:
            env_ids = slice(None)

        object_nodal_positions = dict()

        deformable_object_states = self.scene.get_state(is_relative=True)["deformable_object"]

        if deformable_object_states:
            for obj_name, obj_state in deformable_object_states.items():
                object_nodal_positions[obj_name] = obj_state["nodal_position"][env_ids]

        return object_nodal_positions

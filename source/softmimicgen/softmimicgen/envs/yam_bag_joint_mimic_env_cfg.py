# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.utils import configclass

from softmimicgen.envs.soft_mimic_env_cfg import SoftSubTaskConfig as SubTaskConfig
from softmimicgen_tasks.yam_bag.config.joint_pos_env_cfg import YamBagEnvCfg


@configclass
class YamBagMimicEnvCfg(YamBagEnvCfg, MimicEnvCfg):
    """Isaac Lab Mimic environment config class for YAM Bag with Joint Position control."""

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Override the existing values for data generation
        self.datagen_config.name = "yam_bag_task"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # Subtask configurations for robot0
        subtask_configs_robot0 = []
        subtask_configs_robot0.append(
            SubTaskConfig(
                object_ref="banana",
                object_soft=False,
                subtask_term_signal="wait",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs_robot0.append(
            SubTaskConfig(
                object_ref="banana",
                subtask_term_signal="grasp",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs_robot0.append(
            SubTaskConfig(
                object_ref="bag",
                object_soft=True,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="registration_cost",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["robot0"] = subtask_configs_robot0

        # Subtask configurations for robot1
        subtask_configs_robot1 = []
        subtask_configs_robot1.append(
            SubTaskConfig(
                object_ref="bag",
                object_soft=True,
                subtask_term_signal="lift",
                subtask_term_offset_range=(5, 10),
                selection_strategy="registration_cost",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs_robot1.append(
            SubTaskConfig(
                object_ref="bag",
                object_soft=True,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="registration_cost",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["robot1"] = subtask_configs_robot1

# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.utils import configclass

from softmimicgen.envs.soft_mimic_env_cfg import SoftSubTaskConfig as SubTaskConfig
from softmimicgen_tasks.franka_towel.config.ik_rel_env_cfg import FrankaTowelEnvCfg


@configclass
class FrankaTowelMimicEnvCfg(FrankaTowelEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Franka Towel IK Rel env.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "franka_towel_task"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                object_soft=True,
                subtask_term_signal="grasp_1",
                subtask_term_offset_range=(4, 8),
                selection_strategy="registration_cost",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.01,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                object_soft=True,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="registration_cost",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.01,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["robot0"] = subtask_configs

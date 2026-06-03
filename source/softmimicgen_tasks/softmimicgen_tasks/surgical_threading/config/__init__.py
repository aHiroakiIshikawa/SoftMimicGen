# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os

import gymnasium as gym

from . import agents, ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Thread-PSM-v0",
    entry_point="softmimicgen.envs.soft_mimic_env:SoftManagerBasedRLMimicEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.SurgicalThreadingEnvCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Thread-PSM-IK-Abs-v0",
    entry_point="softmimicgen.envs.soft_mimic_env:SoftManagerBasedRLMimicEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.SurgicalThreadingEnvCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Thread-PSM-IK-Rel-v0",
    entry_point="softmimicgen.envs.soft_mimic_env:SoftManagerBasedRLMimicEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.SurgicalThreadingEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image.json"),
    },
    disable_env_checker=True,
)

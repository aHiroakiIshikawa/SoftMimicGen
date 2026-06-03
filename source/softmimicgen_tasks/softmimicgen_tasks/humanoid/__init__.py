# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import os

from . import agents, humanoid_teddy_env_cfg, humanoid_towel_env_cfg

gym.register(
    id="Isaac-Teddy-GR1T2-Abs-v0",
    entry_point="softmimicgen.envs.soft_mimic_env:SoftManagerBasedRLMimicEnv",
    kwargs={
        "env_cfg_entry_point": humanoid_teddy_env_cfg.GR1T2TeddyEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_teddy.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Towel-GR1T2-Abs-v0",
    entry_point="softmimicgen.envs.soft_mimic_env:SoftManagerBasedRLMimicEnv",
    kwargs={
        "env_cfg_entry_point": humanoid_towel_env_cfg.GR1T2TowelEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_towel.json"),
    },
    disable_env_checker=True,
)

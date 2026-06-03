# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sub-package with environment wrappers for Isaac Lab Mimic."""

import gymnasium as gym

from .humanoid_teddy_mimic_env import GR1T2TeddyMimicEnv
from .humanoid_teddy_mimic_env_cfg import GR1T2TeddyMimicEnvCfg

from .humanoid_towel_mimic_env import GR1T2TowelMimicEnv
from .humanoid_towel_mimic_env_cfg import GR1T2TowelMimicEnvCfg


gym.register(
    id="Isaac-Teddy-GR1T2-Abs-Mimic-v0",    
    entry_point="softmimicgen.envs.pinocchio_envs:GR1T2TeddyMimicEnv",
    kwargs={
        "env_cfg_entry_point": humanoid_teddy_mimic_env_cfg.GR1T2TeddyMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Towel-GR1T2-Abs-Mimic-v0",    
    entry_point="softmimicgen.envs.pinocchio_envs:GR1T2TowelMimicEnv",
    kwargs={
        "env_cfg_entry_point": humanoid_towel_mimic_env_cfg.GR1T2TowelMimicEnvCfg,
    },
    disable_env_checker=True,
)

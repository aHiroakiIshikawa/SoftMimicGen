# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import os

from . import agents

##
# Register Gym environments.
##

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Fold-Towel-Franka-IK-Rel-v0",
    entry_point="softmimicgen.envs.soft_mimic_env:SoftManagerBasedRLMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:FrankaTowelEnvCfg",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image.json"),
    },
    disable_env_checker=True,
)

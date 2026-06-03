# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extended configurations with deformable object support."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig  # noqa: F401
from isaaclab.utils import configclass


@configclass
class SoftSubTaskConfig(SubTaskConfig):
    """Subtask configuration extended with deformable object support."""

    object_soft: bool = False
    """Whether the reference object is a deformable body."""


@configclass
class SoftManagerBasedRLEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration extended with deformable settling support."""

    settling_steps: int = 0
    """Number of settling iterations after reset. Each iteration runs decimation physics steps. 0 disables settling."""

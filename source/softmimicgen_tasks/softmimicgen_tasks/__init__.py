# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package containing SoftMimicGen task environments for deformable object manipulation."""

import os
import toml

# Conveniences to other module directories via relative paths
SOFTMIMICGEN_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

SOFTMIMICGEN_TASKS_METADATA = toml.load(os.path.join(SOFTMIMICGEN_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = SOFTMIMICGEN_TASKS_METADATA["package"]["version"]

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp", "humanoid"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

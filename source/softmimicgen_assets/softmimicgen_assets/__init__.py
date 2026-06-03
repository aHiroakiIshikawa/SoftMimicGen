# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package containing robot and asset configurations for SoftMimicGen."""

import os
import toml

# Conveniences to other module directories via relative paths
SOFTMIMICGEN_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

SOFTMIMICGEN_ASSETS_DATA_DIR = os.path.join(SOFTMIMICGEN_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

SOFTMIMICGEN_ASSETS_METADATA = toml.load(os.path.join(SOFTMIMICGEN_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = SOFTMIMICGEN_ASSETS_METADATA["package"]["version"]

from .robots import *

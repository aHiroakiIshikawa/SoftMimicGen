# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package containing SoftMimicGen data generation pipeline for deformable object manipulation."""

import os
import toml

# Conveniences to other module directories via relative paths
SOFTMIMICGEN_MIMIC_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

SOFTMIMICGEN_MIMIC_METADATA = toml.load(os.path.join(SOFTMIMICGEN_MIMIC_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = SOFTMIMICGEN_MIMIC_METADATA["package"]["version"]

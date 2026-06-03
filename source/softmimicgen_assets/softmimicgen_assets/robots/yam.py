# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the YAM robots.

The following configurations are available:

* :obj:`YAM_CONFIG`: Single YAM robot arm with gripper
* :obj:`YAM_CONFIG_HIGH_PD_CFG`: Single YAM robot arm with gripper with stiffer PD control
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from softmimicgen_assets import SOFTMIMICGEN_ASSETS_DATA_DIR

##
# Configuration
##

YAM_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{SOFTMIMICGEN_ASSETS_DATA_DIR}/Robots/yam/yam/yam.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=100.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=16, 
            solver_velocity_iteration_count=1
        ),
    ),
    articulation_root_prim_path="/arm",
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "left_finger": 0.04,
            "right_finger": -0.04,
        },
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "yam_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"],
            effort_limit_sim=28.0,
            velocity_limit_sim=100.0,
            stiffness=40.0,
            damping=2.5,
        ),
        "yam_forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint[4-6]"],
            effort_limit_sim=10.0,
            velocity_limit_sim=100.0,
            stiffness=10.0,
            damping=1.0,
        ),
        "yam_gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_finger", "right_finger"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=100.0,
            damping=10.0,
        ),
    },
)
"""Configuration of YAM robot."""

YAM_CONFIG_HIGH_PD_CFG = YAM_CONFIG.copy()
YAM_CONFIG_HIGH_PD_CFG.actuators["yam_shoulder"].stiffness = 800.0
YAM_CONFIG_HIGH_PD_CFG.actuators["yam_shoulder"].damping = 10.0
YAM_CONFIG_HIGH_PD_CFG.actuators["yam_forearm"].stiffness = 800.0
YAM_CONFIG_HIGH_PD_CFG.actuators["yam_forearm"].damping = 10.0
YAM_CONFIG_HIGH_PD_CFG.actuators["yam_gripper"].stiffness = 2e3
YAM_CONFIG_HIGH_PD_CFG.actuators["yam_gripper"].damping = 1e2
YAM_CONFIG_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
"""Configuration of YAM robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""

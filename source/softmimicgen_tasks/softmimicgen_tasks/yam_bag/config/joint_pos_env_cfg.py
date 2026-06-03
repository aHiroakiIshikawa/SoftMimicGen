# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaaclab.assets import DeformableCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from softmimicgen_assets import SOFTMIMICGEN_ASSETS_DATA_DIR

from softmimicgen_tasks.yam_bag import mdp
from softmimicgen_tasks.yam_bag.env_cfg import EnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from softmimicgen_assets.robots.yam import YAM_CONFIG_HIGH_PD_CFG  # isort: skip

@configclass
class YamBagEnvCfg(EnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set robot
        self.scene.robot_1 = YAM_CONFIG_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        self.scene.robot_1.init_state.pos = (0.2525, 0.28, 0.755)
        self.scene.robot_1.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot_2 = YAM_CONFIG_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
        self.scene.robot_2.init_state.pos = (0.2525, -0.33, 0.755)
        self.scene.robot_2.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # Set actions for the specific robot type
        self.actions.arm_1_action = mdp.JointPositionActionCfg(
            asset_name="robot_1", joint_names=["joint.*"], scale=1, use_default_offset=True
        )
        self.actions.arm_2_action = mdp.JointPositionActionCfg(
            asset_name="robot_2", joint_names=["joint.*"], scale=1, use_default_offset=True
        )

        self.actions.gripper_1_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["left_finger", "right_finger"],
            open_command_expr={"left_finger": 0.04, "right_finger": -0.04},
            close_command_expr={"left_finger": 0.0, "right_finger": 0.0},
        )
        self.actions.gripper_2_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_2",
            joint_names=["left_finger", "right_finger"],
            open_command_expr={"left_finger": 0.04, "right_finger": -0.04},
            close_command_expr={"left_finger": 0.0, "right_finger": 0.0},
        )

        rigid_body_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        self.scene.banana = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Banana",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.38, 0.03, 0.785], rot=[0.5, 0, 0, 0.8660254]),
            spawn=UsdFileCfg(
                usd_path=f"{SOFTMIMICGEN_ASSETS_DATA_DIR}/Props/Banana/usd/Banana.usd",
                scale=(1.6, 1.3, 1.0),
                rigid_props=rigid_body_properties,
            ),
        )

        self.scene.bag = DeformableCfg(
            prim_path="{ENV_REGEX_NS}/Bag",
            init_state=DeformableCfg.InitialStateCfg(pos=[0.7, 0.00, 0.84], rot=[0.9848078, 0, 0, 0.1736482]),
            spawn=UsdFileCfg(
                usd_path=f"{SOFTMIMICGEN_ASSETS_DATA_DIR}/Props/Bag/Bag.usd",
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_1_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/arm/arm",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/arm/link_6",
                    name="end_effector",
                ),
            ],
        )
        self.scene.ee_2_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_2/arm/arm",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_2/arm/link_6",
                    name="end_effector",
                ),
            ],
        )

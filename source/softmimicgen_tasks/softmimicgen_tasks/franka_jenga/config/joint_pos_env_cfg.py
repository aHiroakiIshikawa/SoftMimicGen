# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaaclab.assets import DeformableObjectCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from softmimicgen_assets import SOFTMIMICGEN_ASSETS_DATA_DIR

from softmimicgen_tasks.franka_jenga import mdp
from softmimicgen_tasks.franka_jenga.env_cfg import EnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaJengaEnvCfg(EnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.usd_path = f"{SOFTMIMICGEN_ASSETS_DATA_DIR}/Robots/FrankaEmika_Bracket/panda_instanceable_bracket.usd"

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.02)),
            spawn=UsdFileCfg(
                usd_path=f"{SOFTMIMICGEN_ASSETS_DATA_DIR}/Props/Rope/Rope_Jenga.usd",
                scale=(0.4, 1.0, 1.0),
            ),
            debug_vis=False,
        )
        self.scene.object.visualizer_cfg.markers["target"].radius = 0.002

        self.scene.jenga = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Jenga",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.83, 0.0, 0.1512), rot=(0.70711, 0.0, 0.0, 0.70711)),
            spawn=UsdFileCfg(
                usd_path=f"{SOFTMIMICGEN_ASSETS_DATA_DIR}/Props/Jenga/Jenga.usd",
                scale=(0.14, 0.14, 0.14),
            ),
        )

        self.scene.jenga_block = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Jenga_Block",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.83, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{SOFTMIMICGEN_ASSETS_DATA_DIR}/Props/Jenga/Jenga_Block.usd",
                scale=(0.14, 0.14, 0.14),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        # Disable replicate physics as it doesn't work for deformable objects
        self.scene.replicate_physics = False

# Copyright (c) 2024-2025, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaaclab.assets import DeformableObjectCfg, RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from softmimicgen_tasks.surgical_threading import mdp
from softmimicgen_tasks.surgical_threading.env_cfg import EnvCfg
from softmimicgen_assets import SOFTMIMICGEN_ASSETS_DATA_DIR

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from softmimicgen_assets.robots.psm_forceps import PSM_CFG  # isort: skip


@configclass
class SurgicalThreadingEnvCfg(EnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set PSM as robot
        self.scene.robot = PSM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (-0.07, 0.0, 0.15)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # Set actions for the specific robot type (PSM)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr={"psm_tool_gripper1_joint": -0.5, "psm_tool_gripper2_joint": 0.5},
            close_command_expr={"psm_tool_gripper1_joint": -0.06, "psm_tool_gripper2_joint": 0.06},
        )

        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(-0.1, 0.0, 0.01), rot=(1.0, 0, 0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{SOFTMIMICGEN_ASSETS_DATA_DIR}/Props/Rope/Rope.usd",
                scale=(0.2, 0.2, 0.2),
            ),
            debug_vis=False,
        )
        self.scene.object.visualizer_cfg.markers["target"].radius = 0.002

        self.scene.ring = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ring",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.03)),
            spawn=UsdFileCfg(
                usd_path=f"{SOFTMIMICGEN_ASSETS_DATA_DIR}/Props/Ring/Ring.usd",
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/psm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/psm_tool_tip_link",
                    name="end_effector",
                ),
            ],
        )

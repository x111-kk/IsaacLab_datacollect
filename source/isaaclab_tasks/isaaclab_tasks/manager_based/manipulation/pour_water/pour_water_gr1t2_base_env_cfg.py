# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for the GR1T2 pour-water manipulation task.

The task: pour the contents (a small "water" proxy object) from a *source* cup
into a *target* cup. Both cups are randomized in position and yaw at every
reset, and the water proxy starts inside the source cup.

The scene is intentionally close to the existing ``nutpour_gr1t2_base_env_cfg``
so the same Pink-IK + Mimic data-augmentation pipeline can be reused. The main
differences are:

* Only two containers (a source cup and a target cup) — no scale/bin.
* The water proxy is a small green ball that starts in the source cup.
* Both cups are randomized at reset (position and yaw).
"""

import tempfile
from dataclasses import MISSING

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg, SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from . import mdp

from isaaclab_assets.robots.fourier import GR1T2_CFG  # isort: skip


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the GR1T2 Pour Water Base Scene."""

    # Table
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/table.usd",
            scale=(1.0, 1.0, 1.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # Source cup -- the cup the robot picks up and pours from.
    # Reuses the ``sorting_beaker_red`` USD as a generic narrow cup.
    source_cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SourceCup",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.13739, 0.45793, 0.9861], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_beaker_red.usd",
            scale=(0.45, 0.45, 1.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # Target cup -- the cup the robot pours into.
    # Reuses the wider ``sorting_bowl_yellow`` USD so the pour is achievable.
    target_cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCup",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.02779, 0.43007, 0.9860], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bowl_yellow.usd",
            scale=(1.0, 1.0, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
        ),
    )

    # Water proxy -- a single small rigid object representing the contents to
    # be poured. Reuses the factory nut USD (small, distinct color) and is
    # initialized inside the source cup at reset time.
    water = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Water",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.13739, 0.45793, 0.9995], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/factory_m16_nut_green.usd",
            scale=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
        ),
    )

    robot: ArticulationCfg = GR1T2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.93),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.5708,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": -1.5708,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # right hand
                "R_index_intermediate_joint": 0.0,
                "R_index_proximal_joint": 0.0,
                "R_middle_intermediate_joint": 0.0,
                "R_middle_proximal_joint": 0.0,
                "R_pinky_intermediate_joint": 0.0,
                "R_pinky_proximal_joint": 0.0,
                "R_ring_intermediate_joint": 0.0,
                "R_ring_proximal_joint": 0.0,
                "R_thumb_distal_joint": 0.0,
                "R_thumb_proximal_pitch_joint": 0.0,
                "R_thumb_proximal_yaw_joint": -1.57,
                # left hand
                "L_index_intermediate_joint": 0.0,
                "L_index_proximal_joint": 0.0,
                "L_middle_intermediate_joint": 0.0,
                "L_middle_proximal_joint": 0.0,
                "L_pinky_intermediate_joint": 0.0,
                "L_pinky_proximal_joint": 0.0,
                "L_ring_intermediate_joint": 0.0,
                "L_ring_proximal_joint": 0.0,
                "L_thumb_distal_joint": 0.0,
                "L_thumb_proximal_pitch_joint": 0.0,
                "L_thumb_proximal_yaw_joint": -1.57,
                # --
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Set table view camera
    robot_pov_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RobotPOVCam",
        update_period=0.0,
        height=160,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.12, 1.67675), rot=(-0.19848, 0.9801, 0.0, 0.0), convention="ros"),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    gr1_action: ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "left_hand_roll_link"})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "left_hand_roll_link"})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "right_hand_roll_link"})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "right_hand_roll_link"})

        hand_joint_state = ObsTerm(func=mdp.get_robot_joint_state, params={"joint_names": ["R_.*", "L_.*"]})
        head_joint_state = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]},
        )

        robot_pov_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("robot_pov_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    source_cup_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("source_cup")},
    )
    target_cup_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("target_cup")},
    )
    water_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("water")},
    )

    success = DoneTerm(func=mdp.task_done_pour_water)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Randomize the position and yaw of *both* cups at every reset.
    # The water proxy is moved together with the source cup so that it stays
    # inside the source cup at reset time.
    reset_object = EventTerm(
        func=mdp.reset_object_poses_pour_water,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.06, 0.06),
                "y": (-0.06, 0.06),
                "yaw": (-0.3, 0.3),
            },
        },
    )


@configclass
class PourWaterGR1T2BaseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 Pour Water environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    # OpenXR hand tracking has 26 joints per hand
    NUM_OPENXR_HAND_JOINTS = 26

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Idle action to hold robot in default pose
    # Action format: [left arm pos (3), left arm quat (4), right arm pos (3),
    #                 right arm quat (4), left/right hand joint pos (22)]
    idle_action = torch.tensor(
        [
            [
                -0.22878,
                0.2536,
                1.0953,
                0.5,
                0.5,
                -0.5,
                0.5,
                0.22878,
                0.2536,
                1.0953,
                0.5,
                0.5,
                -0.5,
                0.5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ]
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 100
        self.sim.render_interval = 2

        # Set settings for camera rendering
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"  # Use DLAA for higher quality rendering

        # List of image observations in policy observations
        self.image_obs_list = ["robot_pov_cam"]

# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab Mimic configuration for the GR1T2 pour-water task.

Subtask layout:

* Right arm (which carries out the pour):
    1. ``idle_right``           — start, idle pose (object_ref = ``source_cup``)
    2. ``grasp_right``          — pick up the source cup (object_ref = ``source_cup``)
    3. ``approach_target_right``— move the source cup above the target cup
       (object_ref = ``target_cup``)
    4. ``pour_right``           — tilt the source cup to pour the water
       (object_ref = ``target_cup``)
    5. (terminal)               — return the source cup upright and place it
       back (object_ref = ``source_cup``)

* Left arm: not used. A single trivial subtask is registered so the Mimic
  bookkeeping does not break.
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pour_water.pour_water_gr1t2_pink_ik_env_cfg import (
    PourWaterGR1T2PinkIKEnvCfg,
)


@configclass
class PourWaterGR1T2MimicEnvCfg(PourWaterGR1T2PinkIKEnvCfg, MimicEnvCfg):
    """Configuration for the GR1T2 Pour Water Mimic environment."""

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "gr1t2_pour_water_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 1000
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.num_demo_to_render = 10
        self.datagen_config.num_fail_demo_to_render = 25
        self.datagen_config.seed = 10

        # ---- Right arm: full pour pipeline -----------------------------
        right_subtasks: list[SubTaskConfig] = []
        right_subtasks.append(
            SubTaskConfig(
                # Initial idle subtask, anchored to the source cup so the
                # approach trajectory can be reused at any randomized start.
                object_ref="source_cup",
                subtask_term_signal="idle_right",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        right_subtasks.append(
            SubTaskConfig(
                # Grasp the source cup.
                object_ref="source_cup",
                subtask_term_signal="grasp_right",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        right_subtasks.append(
            SubTaskConfig(
                # Carry the source cup over the target cup. Anchored to the
                # *target* cup so this segment generalises across target poses.
                object_ref="target_cup",
                subtask_term_signal="approach_target_right",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        right_subtasks.append(
            SubTaskConfig(
                # Tilt the source cup to pour. Still anchored to the target
                # cup so the pour orientation is preserved relative to it.
                object_ref="target_cup",
                subtask_term_signal="pour_right",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        right_subtasks.append(
            SubTaskConfig(
                # Place the source cup back upright. Final segment of the
                # right-arm trajectory; anchored to the source cup's start
                # pose so the placement generalises to its randomized origin.
                object_ref="source_cup",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["right"] = right_subtasks

        # ---- Left arm: idle ---------------------------------------------
        # The left arm is not actively used. A single subtask is still
        # registered so the Mimic per-arm bookkeeping has an entry.
        left_subtasks: list[SubTaskConfig] = []
        left_subtasks.append(
            SubTaskConfig(
                object_ref="source_cup",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["left"] = left_subtasks

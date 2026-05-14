# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab Mimic configuration for the GR1T2 bimanual pour-water task.

Subtask layout (signals are all *object-based*, computed in
:mod:`isaaclab_tasks.manager_based.manipulation.pour_water.mdp.observations`):

* Right arm (pour pipeline):
    1. ``grasp_source_cup_right``   -- pick up the source cup
       (object_ref = ``source_cup``)
    2. ``bottle_above_target_cup``  -- carry the bottle above the target cup
       (object_ref = ``target_cup``)
    3. ``pour_completed``           -- tilt the bottle past 45 deg and hold the
       mouth over the target cup for ~2 s
       (object_ref = ``target_cup``)
    4. (terminal)                   -- release / return the bottle
       (object_ref = ``source_cup``)

* Left arm (placement pipeline -- runs after the pour):
    1. ``grasp_target_cup_left``    -- pick up the target cup
       (object_ref = ``target_cup``)
    2. (terminal)                   -- place the target cup on the placement
       zone (object_ref = ``placement_zone``); ends when ``target_cup_placed``
       has held for ~1 s (also the overall task-success signal)
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pour_water.pour_water_gr1t2_pink_ik_env_cfg import (
    PourWaterGR1T2PinkIKEnvCfg,
)


@configclass
class PourWaterGR1T2MimicEnvCfg(PourWaterGR1T2PinkIKEnvCfg, MimicEnvCfg):
    """Configuration for the GR1T2 bimanual pour-water Mimic environment."""

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

        # ---- Right arm: grasp bottle -> bottle above cup -> pour ---------
        right_subtasks: list[SubTaskConfig] = []
        right_subtasks.append(
            SubTaskConfig(
                # Right hand grasps the source cup (bottle).
                object_ref="source_cup",
                subtask_term_signal="grasp_source_cup_right",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(10, 20),
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
                # Bottle mouth aligned above target cup. Anchored to target_cup
                # so this segment generalises across target-cup positions.
                object_ref="target_cup",
                subtask_term_signal="bottle_above_target_cup",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(10, 20),
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
                # Pour: tilt the bottle and hold the mouth over the target cup
                # for sustain_steps. Still anchored to target_cup.
                object_ref="target_cup",
                subtask_term_signal="pour_completed",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(10, 20),
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
                # Terminal: release / return the bottle to a stable upright
                # pose. No term signal; runs until the overall task ends.
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

        # ---- Left arm: grasp target cup -> place on zone -----------------
        left_subtasks: list[SubTaskConfig] = []
        left_subtasks.append(
            SubTaskConfig(
                # Left hand grasps the target cup.
                object_ref="target_cup",
                subtask_term_signal="grasp_target_cup_left",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        left_subtasks.append(
            SubTaskConfig(
                # Terminal: place the target cup on the placement zone.
                # Anchored to placement_zone so it generalises across zone
                # randomization. Episode ends when ``target_cup_placed`` (also
                # the task-success signal) has held for the sustain window.
                object_ref="placement_zone",
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

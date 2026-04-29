# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination signals for the GR1T2 pour-water manipulation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done_pour_water(
    env: ManagerBasedRLEnv,
    source_cup_cfg: SceneEntityCfg = SceneEntityCfg("source_cup"),
    target_cup_cfg: SceneEntityCfg = SceneEntityCfg("target_cup"),
    water_cfg: SceneEntityCfg = SceneEntityCfg("water"),
    max_water_to_target_xy: float = 0.06,
    max_water_to_target_z: float = 0.05,
    max_source_cup_tilt_dot: float = 0.85,
    max_water_speed: float = 0.20,
) -> torch.Tensor:
    """Determine if the pour-water task is complete.

    Success conditions (all must be true at the same time):

    1. The water proxy is inside the target cup
       (its xy position is within ``max_water_to_target_xy`` of the target-cup
       xy position, and its z is within ``max_water_to_target_z`` above the
       target-cup z).
    2. The source cup has been returned roughly upright
       (the dot product between its body-z axis and the world-z axis is
       greater than ``max_source_cup_tilt_dot``; 1.0 means perfectly upright).
    3. The water proxy has come to rest
       (its linear speed is below ``max_water_speed``).

    Args:
        env: The RL environment instance.
        source_cup_cfg: Configuration for the source-cup entity.
        target_cup_cfg: Configuration for the target-cup entity.
        water_cfg: Configuration for the water-proxy entity.
        max_water_to_target_xy: Maximum xy distance from the water to the
            target cup centre to count as "in the cup".
        max_water_to_target_z: Maximum z offset from the water to the target
            cup centre to count as "in the cup".
        max_source_cup_tilt_dot: Minimum dot product of the source cup's body-z
            axis with world-z required to call the cup "upright".
        max_water_speed: Maximum linear speed of the water proxy required to
            call the pour "settled".

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    source_cup: RigidObject = env.scene[source_cup_cfg.name]
    target_cup: RigidObject = env.scene[target_cup_cfg.name]
    water: RigidObject = env.scene[water_cfg.name]

    water_pos = water.data.root_pos_w - env.scene.env_origins
    target_cup_pos = target_cup.data.root_pos_w - env.scene.env_origins

    water_to_target_xy = torch.linalg.norm(water_pos[:, :2] - target_cup_pos[:, :2], dim=-1)
    water_to_target_z = torch.abs(water_pos[:, 2] - target_cup_pos[:, 2])

    # Source cup body-z axis projected onto world-z. The 3rd column of the
    # rotation matrix gives the body-z axis expressed in world frame; we read
    # it via the quaternion as 1 - 2*(qx**2 + qy**2). This avoids constructing
    # the full rotation matrix.
    quat = source_cup.data.root_quat_w  # (N, 4) wxyz
    qx = quat[:, 1]
    qy = quat[:, 2]
    body_z_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    water_speed = torch.linalg.norm(water.data.root_lin_vel_w, dim=-1)

    done = water_to_target_xy < max_water_to_target_xy
    done = torch.logical_and(done, water_to_target_z < max_water_to_target_z)
    done = torch.logical_and(done, body_z_world_z > max_source_cup_tilt_dot)
    done = torch.logical_and(done, water_speed < max_water_speed)
    return done

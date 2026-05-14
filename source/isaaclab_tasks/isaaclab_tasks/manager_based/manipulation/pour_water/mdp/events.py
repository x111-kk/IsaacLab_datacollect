# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the GR1T2 pour-water manipulation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_object_poses_pour_water(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cup_pose_range: dict[str, tuple[float, float]],
    zone_pose_range: dict[str, tuple[float, float]],
    source_cup_cfg: SceneEntityCfg = SceneEntityCfg("source_cup"),
    target_cup_cfg: SceneEntityCfg = SceneEntityCfg("target_cup"),
    placement_zone_cfg: SceneEntityCfg = SceneEntityCfg("placement_zone"),
):
    """Reset the source cup, target cup, and placement zone to randomized poses.

    The source and target cups are sampled from ``cup_pose_range`` (independent
    samples so they can each end up anywhere in the rectangle). The placement
    zone is sampled from a separate ``zone_pose_range`` to keep it on its own
    side of the table.

    Args:
        env: The RL environment instance.
        env_ids: The environment IDs to reset the object poses for.
        cup_pose_range: Per-axis ``(low, high)`` ranges for the source and
            target cup randomization. Keys are ``x``, ``y``, ``z``, ``roll``,
            ``pitch``, ``yaw``.
        zone_pose_range: Per-axis ``(low, high)`` ranges for the placement
            zone. Same key conventions as ``cup_pose_range``.
        source_cup_cfg: Configuration for the source-cup asset.
        target_cup_cfg: Configuration for the target-cup asset.
        placement_zone_cfg: Configuration for the placement-zone marker.
    """
    source_cup = env.scene[source_cup_cfg.name]
    target_cup = env.scene[target_cup_cfg.name]
    placement_zone = env.scene[placement_zone_cfg.name]

    source_cup_root_states = source_cup.data.default_root_state[env_ids].clone()
    target_cup_root_states = target_cup.data.default_root_state[env_ids].clone()
    zone_root_states = placement_zone.data.default_root_state[env_ids].clone()

    cup_range_list = [cup_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    cup_ranges = torch.tensor(cup_range_list, device=source_cup.device)
    zone_range_list = [zone_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    zone_ranges = torch.tensor(zone_range_list, device=source_cup.device)

    # Source cup.
    rand_samples = math_utils.sample_uniform(
        cup_ranges[:, 0], cup_ranges[:, 1], (len(env_ids), 6), device=source_cup.device
    )
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    positions_source_cup = source_cup_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_source_cup = math_utils.quat_mul(source_cup_root_states[:, 3:7], orientations_delta)

    # Target cup.
    rand_samples = math_utils.sample_uniform(
        cup_ranges[:, 0], cup_ranges[:, 1], (len(env_ids), 6), device=source_cup.device
    )
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    positions_target_cup = target_cup_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_target_cup = math_utils.quat_mul(target_cup_root_states[:, 3:7], orientations_delta)

    # Placement zone (independent range, typically smaller and offset to one
    # side of the table).
    rand_samples = math_utils.sample_uniform(
        zone_ranges[:, 0], zone_ranges[:, 1], (len(env_ids), 6), device=source_cup.device
    )
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    positions_zone = zone_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_zone = math_utils.quat_mul(zone_root_states[:, 3:7], orientations_delta)

    # Write back to sim.
    source_cup.write_root_pose_to_sim(
        torch.cat([positions_source_cup, orientations_source_cup], dim=-1), env_ids=env_ids
    )
    target_cup.write_root_pose_to_sim(
        torch.cat([positions_target_cup, orientations_target_cup], dim=-1), env_ids=env_ids
    )
    placement_zone.write_root_pose_to_sim(torch.cat([positions_zone, orientations_zone], dim=-1), env_ids=env_ids)

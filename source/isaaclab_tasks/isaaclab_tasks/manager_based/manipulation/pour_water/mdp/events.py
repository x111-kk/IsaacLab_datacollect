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
    pose_range: dict[str, tuple[float, float]],
    source_cup_cfg: SceneEntityCfg = SceneEntityCfg("source_cup"),
    target_cup_cfg: SceneEntityCfg = SceneEntityCfg("target_cup"),
    water_cfg: SceneEntityCfg = SceneEntityCfg("water"),
):
    """Reset the source cup, target cup, and water proxy to randomized poses.

    Both cups are sampled independently from ``pose_range`` (so they can each
    end up anywhere in the configured rectangle). The water proxy is rigidly
    attached to the source-cup randomization so it stays inside the source
    cup at reset time.

    Args:
        env: The RL environment instance.
        env_ids: The environment IDs to reset the object poses for.
        pose_range: The dictionary of pose ranges for the objects. Keys are
            ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.
        source_cup_cfg: The configuration for the source-cup asset.
        target_cup_cfg: The configuration for the target-cup asset.
        water_cfg: The configuration for the water-proxy asset.
    """
    source_cup = env.scene[source_cup_cfg.name]
    target_cup = env.scene[target_cup_cfg.name]
    water = env.scene[water_cfg.name]

    source_cup_root_states = source_cup.data.default_root_state[env_ids].clone()
    target_cup_root_states = target_cup.data.default_root_state[env_ids].clone()
    water_root_states = water.data.default_root_state[env_ids].clone()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=source_cup.device)

    # Randomize the source cup; move the water by the same delta so it stays
    # inside the cup.
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=source_cup.device)
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    positions_source_cup = source_cup_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    positions_water = water_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_source_cup = math_utils.quat_mul(source_cup_root_states[:, 3:7], orientations_delta)
    orientations_water = math_utils.quat_mul(water_root_states[:, 3:7], orientations_delta)

    # Randomize the target cup independently.
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=source_cup.device)
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    positions_target_cup = target_cup_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_target_cup = math_utils.quat_mul(target_cup_root_states[:, 3:7], orientations_delta)

    # Write back to sim.
    source_cup.write_root_pose_to_sim(
        torch.cat([positions_source_cup, orientations_source_cup], dim=-1), env_ids=env_ids
    )
    target_cup.write_root_pose_to_sim(
        torch.cat([positions_target_cup, orientations_target_cup], dim=-1), env_ids=env_ids
    )
    water.write_root_pose_to_sim(torch.cat([positions_water, orientations_water], dim=-1), env_ids=env_ids)

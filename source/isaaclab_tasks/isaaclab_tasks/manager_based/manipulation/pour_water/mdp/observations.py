# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation helpers for the GR1T2 pour-water manipulation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_eef_pos(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    """Return the world-frame position of ``link_name`` (relative to env origin)."""
    body_pos_w = env.scene["robot"].data.body_pos_w
    eef_idx = env.scene["robot"].data.body_names.index(link_name)
    return body_pos_w[:, eef_idx] - env.scene.env_origins


def get_eef_quat(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    """Return the world-frame quaternion of ``link_name`` (wxyz)."""
    body_quat_w = env.scene["robot"].data.body_quat_w
    eef_idx = env.scene["robot"].data.body_names.index(link_name)
    return body_quat_w[:, eef_idx]


def get_robot_joint_state(env: ManagerBasedRLEnv, joint_names: list[str]) -> torch.Tensor:
    """Return joint positions for the joints matched by ``joint_names`` (regex list)."""
    indexes, _ = env.scene["robot"].find_joints(joint_names)
    indexes = torch.tensor(indexes, dtype=torch.long)
    return env.scene["robot"].data.joint_pos[:, indexes]

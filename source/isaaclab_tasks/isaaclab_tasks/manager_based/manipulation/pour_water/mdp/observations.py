# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation helpers for the GR1T2 pour-water manipulation task.

This module provides two families of observations:

1. Generic robot-state helpers (`get_eef_pos`, `get_eef_quat`, ...).
2. **Subtask termination signals** consumed by Isaac Lab Mimic to splice
   trajectories at meaningful task boundaries. All subtask signals key off
   *object pose* (not end-effector pose) so they remain correct even if the
   grasped object slips inside the hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

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


# ---------------------------------------------------------------------------
# Subtask termination signals (consumed by Isaac Lab Mimic).
# Each signal is a boolean tensor of shape ``(num_envs,)`` exposed via the
# ``subtask_terms`` observation group.
# ---------------------------------------------------------------------------


def _object_lifted(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    min_lift: float,
) -> torch.Tensor:
    """Return ``True`` for envs where ``object`` has been lifted above its default z."""
    obj: RigidObject = env.scene[object_cfg.name]
    default_z = obj.data.default_root_state[:, 2] + env.scene.env_origins[:, 2]
    return (obj.data.root_pos_w[:, 2] - default_z) > min_lift


def _eef_near_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    eef_link: str,
    max_dist: float,
) -> torch.Tensor:
    """Return ``True`` for envs where ``eef_link`` is within ``max_dist`` of ``object``."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot = env.scene["robot"]
    eef_idx = robot.data.body_names.index(eef_link)
    eef_pos_w = robot.data.body_pos_w[:, eef_idx, :]
    dist = torch.linalg.norm(eef_pos_w - obj.data.root_pos_w, dim=-1)
    return dist < max_dist


def _world_offset_from_root(
    obj: RigidObject,
    offset_local: tuple[float, float, float],
) -> torch.Tensor:
    """Return ``obj.root_pos_w + R(obj.root_quat_w) @ offset_local`` per env."""
    quat = obj.data.root_quat_w
    num_envs = quat.shape[0]
    offset = torch.tensor(offset_local, device=quat.device, dtype=torch.float32)
    offset = offset.unsqueeze(0).expand(num_envs, 3).contiguous()
    rotated = math_utils.quat_apply(quat, offset)
    return obj.data.root_pos_w + rotated


def grasp_source_cup_left(
    env: ManagerBasedRLEnv,
    source_cup_cfg: SceneEntityCfg = SceneEntityCfg("source_cup"),
    eef_link: str = "left_hand_pitch_link",
    max_dist: float = 0.10,
    min_lift: float = 0.05,
) -> torch.Tensor:
    """Subtask 1: left hand has grasped the source cup (bottle) and lifted it."""
    near = _eef_near_object(env, source_cup_cfg, eef_link, max_dist)
    lifted = _object_lifted(env, source_cup_cfg, min_lift)
    return near & lifted


def bottle_above_target_cup(
    env: ManagerBasedRLEnv,
    source_cup_cfg: SceneEntityCfg = SceneEntityCfg("source_cup"),
    target_cup_cfg: SceneEntityCfg = SceneEntityCfg("target_cup"),
    bottle_mouth_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.15),
    cup_mouth_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.05),
    max_mouth_xy: float = 0.05,
    min_mouth_z: float = 0.00,
    max_mouth_z: float = 0.20,
) -> torch.Tensor:
    """Subtask 2: bottle mouth is positioned above the target-cup mouth.

    The check is purely *object-based* — neither the end-effector nor the grasp
    geometry are referenced. If the bottle slips inside the hand the signal
    follows the bottle, not the hand.
    """
    source_cup: RigidObject = env.scene[source_cup_cfg.name]
    target_cup: RigidObject = env.scene[target_cup_cfg.name]
    bottle_mouth_w = _world_offset_from_root(source_cup, bottle_mouth_offset_local)
    cup_mouth_w = _world_offset_from_root(target_cup, cup_mouth_offset_local)
    delta = bottle_mouth_w - cup_mouth_w
    xy_dist = torch.linalg.norm(delta[:, :2], dim=-1)
    z_offset = delta[:, 2]
    return (xy_dist < max_mouth_xy) & (z_offset > min_mouth_z) & (z_offset < max_mouth_z)


def pour_completed(
    env: ManagerBasedRLEnv,
    source_cup_cfg: SceneEntityCfg = SceneEntityCfg("source_cup"),
    target_cup_cfg: SceneEntityCfg = SceneEntityCfg("target_cup"),
    tilt_threshold_deg: float = 45.0,
    bottle_mouth_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.15),
    cup_mouth_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.05),
    max_mouth_xy: float = 0.05,
    min_mouth_z: float = 0.00,
    max_mouth_z: float = 0.20,
    sustain_steps: int = 200,
) -> torch.Tensor:
    """Subtask 3: pour has been executed (bottle tilted past threshold and the
    mouth has been kept over the target cup for ``sustain_steps`` consecutive
    ticks).

    Tilt is measured from the *bottle* body-z axis, not the gripper. The sustain
    counter is attached to ``env`` and resets to zero on any frame where the
    condition fails (which naturally happens on env reset, since the bottle is
    randomized back to upright).
    """
    source_cup: RigidObject = env.scene[source_cup_cfg.name]
    target_cup: RigidObject = env.scene[target_cup_cfg.name]

    # Tilt: bottle body-z axis projected on world-z. Uses the
    # ``1 - 2*(qx^2 + qy^2)`` form of the (3,3) rotation-matrix entry.
    quat = source_cup.data.root_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    body_z_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    cos_threshold = torch.cos(torch.tensor(tilt_threshold_deg * torch.pi / 180.0, device=quat.device))
    tilted = body_z_world_z < cos_threshold

    # Mouth alignment (object-based).
    bottle_mouth_w = _world_offset_from_root(source_cup, bottle_mouth_offset_local)
    cup_mouth_w = _world_offset_from_root(target_cup, cup_mouth_offset_local)
    delta = bottle_mouth_w - cup_mouth_w
    xy_dist = torch.linalg.norm(delta[:, :2], dim=-1)
    z_offset = delta[:, 2]
    aligned = (xy_dist < max_mouth_xy) & (z_offset > min_mouth_z) & (z_offset < max_mouth_z)

    cond = tilted & aligned

    # Sustain counter attached to the env. Reset to zero on any failing frame.
    counter_name = "_pour_completed_counter"
    counter = getattr(env, counter_name, None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    counter = torch.where(cond, counter + 1, torch.zeros_like(counter))
    setattr(env, counter_name, counter)

    return counter >= sustain_steps


def grasp_target_cup_right(
    env: ManagerBasedRLEnv,
    target_cup_cfg: SceneEntityCfg = SceneEntityCfg("target_cup"),
    eef_link: str = "right_hand_pitch_link",
    max_dist: float = 0.10,
    min_lift: float = 0.05,
) -> torch.Tensor:
    """Subtask 4: right hand has grasped the target cup and lifted it."""
    near = _eef_near_object(env, target_cup_cfg, eef_link, max_dist)
    lifted = _object_lifted(env, target_cup_cfg, min_lift)
    return near & lifted


def target_cup_placed(
    env: ManagerBasedRLEnv,
    target_cup_cfg: SceneEntityCfg = SceneEntityCfg("target_cup"),
    placement_zone_cfg: SceneEntityCfg = SceneEntityCfg("placement_zone"),
    max_xy: float = 0.04,
    max_z: float = 0.03,
    sustain_steps: int = 100,
) -> torch.Tensor:
    """Subtask 5 (and final task success): the target cup has been placed onto
    the placement zone (xy aligned, z close, and held for ``sustain_steps``).

    Like :func:`pour_completed`, the sustain counter lives on ``env`` and resets
    on any failing frame.
    """
    target_cup: RigidObject = env.scene[target_cup_cfg.name]
    zone: RigidObject = env.scene[placement_zone_cfg.name]
    delta = target_cup.data.root_pos_w - zone.data.root_pos_w
    xy_dist = torch.linalg.norm(delta[:, :2], dim=-1)
    z_dist = torch.abs(delta[:, 2])
    cond = (xy_dist < max_xy) & (z_dist < max_z)

    counter_name = "_target_cup_placed_counter"
    counter = getattr(env, counter_name, None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    counter = torch.where(cond, counter + 1, torch.zeros_like(counter))
    setattr(env, counter_name, counter)

    return counter >= sustain_steps

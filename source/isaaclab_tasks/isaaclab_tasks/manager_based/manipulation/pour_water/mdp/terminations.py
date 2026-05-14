# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination signals for the GR1T2 pour-water manipulation task.

The full pipeline is:

1. Right hand grasps the source cup (bottle).
2. Right hand carries the bottle above the target cup.
3. Right hand tilts the bottle past ``tilt_threshold_deg`` and holds it over
   the target cup for ``sustain_steps`` ticks.
4. Left hand grasps the target cup.
5. Left hand places the target cup onto the placement zone and holds it there
   for ``sustain_steps`` ticks. This is also the **task-success** signal.

All checks key off the *object* poses (bottle / cup / zone), never the
end-effector pose, so they remain correct under in-hand slip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import target_cup_placed

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done_pour_water(
    env: ManagerBasedRLEnv,
    target_cup_cfg: SceneEntityCfg = SceneEntityCfg("target_cup"),
    placement_zone_cfg: SceneEntityCfg = SceneEntityCfg("placement_zone"),
    max_xy: float = 0.04,
    max_z: float = 0.03,
    sustain_steps: int = 100,
) -> torch.Tensor:
    """Final task-success signal for the bimanual pour-water pipeline.

    The full task is considered complete only after the target cup has been
    placed on the placement zone and held there for ``sustain_steps`` ticks
    (the same check used by the
    :func:`~.observations.target_cup_placed` subtask signal).

    Args:
        env: The RL environment instance.
        target_cup_cfg: Configuration for the target-cup entity.
        placement_zone_cfg: Configuration for the placement-zone marker.
        max_xy: Maximum xy distance from the cup to the zone centre.
        max_z: Maximum z distance from the cup to the zone centre (cup is
            considered "on" the zone when both axes are within tolerance).
        sustain_steps: Number of consecutive ticks the placement condition must
            hold before success is declared.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    return target_cup_placed(
        env,
        target_cup_cfg=target_cup_cfg,
        placement_zone_cfg=placement_zone_cfg,
        max_xy=max_xy,
        max_z=max_z,
        sustain_steps=sustain_steps,
    )

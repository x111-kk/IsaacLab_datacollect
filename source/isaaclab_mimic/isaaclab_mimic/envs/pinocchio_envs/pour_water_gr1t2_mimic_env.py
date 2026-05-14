# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mimic environment wrapper for the GR1T2 bimanual pour-water task.

This wrapper inherits the EEF / action / object-pose plumbing from
:class:`PickPlaceGR1T2MimicEnv` and adds an implementation of
:py:meth:`get_subtask_term_signals` that surfaces the per-subtask boolean
signals computed in the ``subtask_terms`` observation group of
``PourWaterGR1T2BaseEnvCfg``.
"""

from collections.abc import Sequence

import torch

from .pickplace_gr1t2_mimic_env import PickPlaceGR1T2MimicEnv


class PourWaterGR1T2MimicEnv(PickPlaceGR1T2MimicEnv):
    """GR1T2 bimanual pour-water Mimic environment wrapper."""

    #: Subtask signal names exposed via the ``subtask_terms`` observation group
    #: in :class:`PourWaterGR1T2BaseEnvCfg`. The order matches the right-then-
    #: left-arm subtask sequence in :class:`PourWaterGR1T2MimicEnvCfg`.
    _SUBTASK_SIGNAL_NAMES: tuple[str, ...] = (
        "grasp_source_cup_right",
        "bottle_above_target_cup",
        "pour_completed",
        "grasp_target_cup_left",
        "target_cup_placed",
    )

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return the per-subtask termination signals consumed by Mimic.

        Args:
            env_ids: Environment indices to gather signals for. If ``None``,
                signals are returned for every env.

        Returns:
            Dictionary mapping subtask names to boolean tensors of shape
            ``(len(env_ids),)``.
        """
        if env_ids is None:
            env_ids = slice(None)

        subtask_terms = self.obs_buf["subtask_terms"]
        return {name: subtask_terms[name][env_ids] for name in self._SUBTASK_SIGNAL_NAMES}

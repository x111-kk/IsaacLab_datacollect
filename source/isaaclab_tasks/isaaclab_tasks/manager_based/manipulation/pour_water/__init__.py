# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR1T2 bimanual pour-water manipulation task.

The robot pours a small water-proxy object from a *source* cup into a *target*
cup. Both cups are randomized in position and yaw at every reset. The task is
intended for human teleoperation data collection (via ``record_demos.py``) and
Isaac Lab Mimic data augmentation.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-PourWater-GR1T2-Pink-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pour_water_gr1t2_pink_ik_env_cfg:PourWaterGR1T2PinkIKEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_pour_water.json",
    },
    disable_env_checker=True,
)

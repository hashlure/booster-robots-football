# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Booster-Liedown-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:LiedownEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:LiedownPPORunnerCfg",
    },
)
gym.register(
    id="Booster-Standup-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:StandupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:StandupPPORunnerCfg",
    },
)

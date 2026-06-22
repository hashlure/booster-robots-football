import gymnasium as gym

gym.register(
    id="Booster-WalkToBall-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

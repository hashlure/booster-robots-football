from isaaclab.utils import configclass
from ..walk2run_amp.ppo_cfg import PPORunnerCfg as Walk2RunPPORunnerCfg


@configclass
class PPORunnerCfg(Walk2RunPPORunnerCfg):
    """走向球任务 — 继承 walk2run_amp，只改数据和实验名。"""

    experiment_name = "walk_to_ball_amp"

    # 只用走/跑参考数据
    amp_motion_files = [
        "/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/walk.txt",
        "/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/run.txt",
        "/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/walk2run.txt",
    ]

    amp_reward_coef = 0.3  # 和原始一致

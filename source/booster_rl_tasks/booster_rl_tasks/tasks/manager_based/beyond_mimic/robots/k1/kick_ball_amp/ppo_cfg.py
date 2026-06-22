from isaaclab.utils import configclass
from ..walk2run_amp.ppo_cfg import PPORunnerCfg as Walk2RunPPORunnerCfg


@configclass
class PPORunnerCfg(Walk2RunPPORunnerCfg):
    """踢球任务 — 继承 walk2run_amp，只改数据和实验名"""

    experiment_name = "kick_ball_amp"

    # =========================================================================
    # AMP 参考数据 — 分阶段
    # Phase 1 (走路靠近): 只用 locomotion，让机器人先学会站稳走路
    # Phase 2/3 (踢球):   取消注释，加入踢球数据
    # =========================================================================
    amp_motion_files = [
        # Phase 1 — locomotion
        "/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/walk.txt",
        "/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/walk2run.txt",
        "/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/run2walk.txt",
        "/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/run.txt",
        #Phase 2/3/4 — 取消注释加入踢球数据（原始 + 镜像，双脚都会踢）
        "/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/kick.txt",
        "/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/kick_mirror.txt",
        # "/root/GVHMR/outputs/demo/kick/kick1/kick1_k1.txt",
        # "/root/GVHMR/outputs/demo/kick/kick1/kick1_k1_mirror.txt",
        # "/root/GVHMR/outputs/demo/kick/kcik2/kcik2_k1.txt",
        # "/root/GVHMR/outputs/demo/kick/kcik2/kcik2_k1_mirror.txt",
        "/root/GVHMR/outputs/demo/kick/kick4/kick4_k1.txt",
        "/root/GVHMR/outputs/demo/kick/kick4/kick4_k1_mirror.txt",
    ]

    # AMP 风格奖励系数
    amp_reward_coef = 0.35

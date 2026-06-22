from isaaclab.utils import configclass
from ...k1.walk2run_amp.ppo_cfg import PPORunnerCfg as Walk2RunPPORunnerCfg

T1_AMASS_DIR = "/root/gpufree-data/GVHMR_outputs/demo/T1/amass"


@configclass
class PPORunnerCfg(Walk2RunPPORunnerCfg):
    """踢球任务 — 继承 walk2run_amp，只改数据和实验名"""

    experiment_name = "kick_ball_amp_t1"

    T1_KICK_DIR = "/root/GVHMR/outputs/demo/T1"
    # T1 AMP 参考数据 — 分阶段
    # Phase 1 (走路靠近): 只用 locomotion
    # Phase 2/3/4 (踢球):  追加踢球数据
    amp_motion_files = [
        # Phase 1 — locomotion
        f"{T1_AMASS_DIR}/07_walk/07_01_stageii.txt",
        f"{T1_AMASS_DIR}/07_walk/07_02_stageii.txt",
        f"{T1_AMASS_DIR}/07_walk/07_04_stageii.txt",
        f"{T1_AMASS_DIR}/09_run/09_04_stageii.txt",
        f"{T1_AMASS_DIR}/09_run/09_10_stageii.txt",
        # Phase 2/3/4 — 取消注释加入踢球数据
        f"{T1_KICK_DIR}/kick1/kick1_t1.txt",
        f"{T1_KICK_DIR}/kick1/kick1_t1_mirror.txt",
        f"{T1_KICK_DIR}/kcik2/kcik2_t1.txt",
        f"{T1_KICK_DIR}/kcik2/kcik2_t1_mirror.txt",
        f"{T1_KICK_DIR}/kick4/kick4_t1.txt",
        f"{T1_KICK_DIR}/kick4/kick4_t1_mirror.txt",
    ]
    min_normalized_std = [0.05] * 23

    # AMP 风格奖励系数
    amp_reward_coef = 0.1

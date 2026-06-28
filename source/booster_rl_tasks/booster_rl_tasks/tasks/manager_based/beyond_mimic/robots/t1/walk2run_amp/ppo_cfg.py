from isaaclab.utils import configclass
from ...k1.walk2run_amp.ppo_cfg import PPORunnerCfg as Walk2RunPPORunnerCfg


@configclass
class PPORunnerCfg(Walk2RunPPORunnerCfg):
    """T1 walk2run — 继承 K1 配置，关 AMP，纯 PPO。"""

    experiment_name = "walk2run_ppo_t1"
    amp_reward_coef = 0.3
    min_normalized_std = [0.05] * 23

    T1_AMASS_DIR = "/root/gpufree-data/GVHMR_outputs/demo/T1/amass"
    amp_motion_files = [
        f"{T1_AMASS_DIR}/07_walk/07_01_stageii.txt",
        f"{T1_AMASS_DIR}/07_walk/07_02_stageii.txt",
        f"{T1_AMASS_DIR}/07_walk/07_03_stageii.txt",
        f"{T1_AMASS_DIR}/07_walk/07_04_stageii.txt",
        f"{T1_AMASS_DIR}/09_run/09_04_stageii.txt",
        f"{T1_AMASS_DIR}/09_run/09_10_stageii.txt",
        f"/root/GVHMR/outputs/demo/T1/amass/35/35_27_stageii.txt",
        f"/root/GVHMR/outputs/demo/T1/amass/16/16_34_stageii.txt"
        # f"{T1_AMASS_DIR}/09_run/09_12_stageii.txt",
    ]

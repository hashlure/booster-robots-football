from isaaclab.utils import configclass
from ...k1.walk2run_amp.ppo_cfg import PPORunnerCfg as Walk2RunPPORunnerCfg


@configclass
class PPORunnerCfg(Walk2RunPPORunnerCfg):
    """T1 walk2run — resume from checkpoint with stand_still + push training."""

    experiment_name = "walk2run_ppo_t1"
    run_name = "stand_finetune"
    resume = True
    load_run = "2026-06-30_14-47-39_stand_finetune"
    load_checkpoint = "model_6000.pt"
    amp_reward_coef = 0.3
    min_normalized_std = [0.05] * 23

    T1_AMASS_DIR = "/root/gpufree-data/GVHMR_outputs/demo/T1/amass"
    amp_motion_files = [
        f"{T1_AMASS_DIR}/07_walk/07_01_stageii.txt",
        f"{T1_AMASS_DIR}/07_walk/07_02_stageii.txt",
        f"/root/gpufree-data/my_video/walk2stand/walk2stand_t1_mirror.txt",
        f"/root/gpufree-data/my_video/walk2stand/walk2stand_t1.txt",
        f"/root/gpufree-data/my_video/stand2walk/stand2walk_t1_mirror.txt",
        f"/root/gpufree-data/my_video/stand2walk/stand2walk_t1.txt",
        f"/root/GVHMR/outputs/demo/T1/amass/35/35_27_stageii.txt",
        f"/root/GVHMR/outputs/demo/T1/amass/16/16_34_stageii.txt",
        f"/root/gpufree-data/my_video/circle/circle_t1.txt",
        f"/root/gpufree-data/my_video/move_aroud/move_aroud_t1.txt",
        f"/root/gpufree-data/my_video/move_y/move_y_t1.txt"
        # f"{T1_AMASS_DIR}/09_run/09_12_stageii.txt",
    ]

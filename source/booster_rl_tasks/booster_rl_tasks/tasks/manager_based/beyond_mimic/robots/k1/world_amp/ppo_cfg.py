from isaaclab.utils import configclass
from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BaseAMPAgentCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg, RslRlRndCfg
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp import symmetry
from dataclasses import MISSING

@configclass
class depth_predictor_cfg:
    lr: float= 3e-4
    weight_decay: float= 1e-4
    training_interval: int= 10
    training_iters: int= 1000
    batch_size: int= 1024
    loss_scale: int= 100
    #  
    resized: tuple= (64, 64)
    update_interval: int= 5  # 5 works without retraining, 8 worse
    camera_num_envs: int = 1024
    forward_height_dim: int = 525 
    use_camera: bool = False

@configclass
class PPORunnerCfg(BaseAMPAgentCfg):
    max_iterations = 50000
    experiment_name = "wm_amp"

    # amp parameter
    amp_reward_coef = 0.3
    amp_motion_files = ["/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/walk.txt","/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/walk2run.txt","/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/run.txt","/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/kick.txt","/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/run2walk.txt"]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 22
    # depth_predictor = depth_predictor_cfg()

    # latent_dim = 35
    # wm_latent_dim = 32
    # wm_encoder_hidden_dims = [64, 64]

    # prop_dim = 78
    # height_dim = 121
    # privileged_dim = 189

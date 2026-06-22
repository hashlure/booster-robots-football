from isaaclab.utils import configclass
from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BaseAMPAgentCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg, RslRlRndCfg
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp import symmetry


@configclass
class PPORunnerCfg(BaseAMPAgentCfg):
    max_iterations = 50000
    experiment_name = "walk2run_amp"

    # amp parameter
    amp_reward_coef = 0.3
    #amp_motion_files = ["/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/walk.txt","/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/walk2run.txt","/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/run.txt","/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/kick.txt","/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/run2walk.txt"]
    amp_motion_files = ["/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/walk.txt","/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/walk2run.txt","/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/run.txt","/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/run2walk.txt"]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 22
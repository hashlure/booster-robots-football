"""T1 deploy-aligned AMP config — 关 empirical normalization，PD 对齐真机。"""

from isaaclab.utils import configclass
from ..walk2run_amp.ppo_cfg import PPORunnerCfg as T1Walk2RunPPORunnerCfg


@configclass
class PPORunnerCfg(T1Walk2RunPPORunnerCfg):
    experiment_name = "walk_deploy2_t1"
    empirical_normalization = False   # 关掉，匹配 deploy 硬编码归一化

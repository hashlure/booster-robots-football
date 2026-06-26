"""T1 walk2run — body name 适配 + sim2real 域随机化。"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from ...k1.walk2run_amp.tracking_env_cfg import (
    RewardsCfg as BaseRewardsCfg,
    TrackingEnvCfg as BaseTrackingEnvCfg,
)
import booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp as mdp


@configclass
class RewardsCfg(BaseRewardsCfg):
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    head_height = RewTerm(func=mdp.tracking_head_height,
        params={"target_head_height": 0.40, "threshold": 0.92, "std": 0.3,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", body_names=["H2"])},
        weight=0.0)
    base_height_l2 = RewTerm(func=mdp.base_height_l2,
        params={"target_height": 0.68}, weight=-5.0)
    keep_balance = RewTerm(func=mdp.stay_alive, weight=0.1)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.7071}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": 0.7071}
    )
    stand_still = RewTerm(func=mdp.stand_still, weight=0.0, params={"command_name": "base_velocity"})
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces",
               body_names=["Shank_.*", "Hip_.*", ".*hand_.*", "AL.*", "AR.*", "H1", "H2", "Trunk"]),
               "threshold": 1.0},
    )


# ============================================================
# Sim2Real 五阶段课程
# ============================================================
# Phase 1 (0-1500):  学走路 — 低速、无转向、无扰动，轻度惩罚
#   env_cfg: ang_vel_z=(0,0), lin_vel_x=(0.3, 0.8)
# Phase 2 (1500-3000): 加转向 — 开启角速度指令
#   env_cfg: ang_vel_z=(-0.3, 0.3), lin_vel_x=(0.5, 1.5)
# Phase 3 (3000-5000): 抗小扰 — 轻度推力，关节偏移
#   取消下面 Phase3 注释块，events: EventCfg3 = EventCfg3()
# Phase 4 (5000-7000): 大速度 — 高速指令，中等扰动
#   env_cfg: lin_vel_x=(1.0, 2.5)
#   取消下面 Phase4 注释块，events: EventCfg4 = EventCfg4()
# Phase 5 (7000+):     抗大扰 — 强随机化 + 强关节惩罚
#   取消下面 Phase5 注释块，events: EventCfg5 = EventCfg5()
#   同时加大 dof_acc, action_rate 惩罚
# ============================================================


@configclass
class TrackingEnvCfg(BaseTrackingEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    # Phase 3-5 按需取消下面注释：
    # events: EventCfg3 = EventCfg3()
    # events: EventCfg4 = EventCfg4()
    # events: EventCfg5 = EventCfg5()

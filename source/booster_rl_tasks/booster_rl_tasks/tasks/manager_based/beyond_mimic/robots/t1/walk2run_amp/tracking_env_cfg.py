"""T1 walk2run — 仅覆盖 K1 版中 body name 不同的奖励项。"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from ...k1.walk2run_amp.tracking_env_cfg import RewardsCfg as BaseRewardsCfg, TrackingEnvCfg as BaseTrackingEnvCfg
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
    # T1 头部高度 → 权重归零（正确站姿给奖励=奖赏不动，改用 base_height 惩罚）
    head_height = RewTerm(func=mdp.tracking_head_height,
        params={"target_head_height": 0.40, "threshold": 0.92, "std": 0.3,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", body_names=["H2"])},
        weight=0.0)
    base_height_l2 = RewTerm(func=mdp.base_height_l2,
        params={"target_height": 0.68}, weight=-5.0)
    # 杀死不动策略
    keep_balance = RewTerm(func=mdp.stay_alive, weight=0.1)
    # 线速度跟踪提到 3.0，成为最大单一正向奖励
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.7071}
    )
    # 关掉角速度跟踪和静止惩罚
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": 0.7071}
    )
    stand_still = RewTerm(func=mdp.stand_still, weight=0.0, params={"command_name": "base_velocity"})
    # T1 不该碰地的部位
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces",
               body_names=["Shank_.*", "Hip_.*", ".*hand_.*", "AL.*", "AR.*", "H1", "H2", "Trunk"]),
               "threshold": 1.0},
    )


@configclass
class TrackingEnvCfg(BaseTrackingEnvCfg):
    rewards: RewardsCfg = RewardsCfg()

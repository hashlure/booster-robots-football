"""T1 walk2run — body name 适配 + 单阶段全开命令空间（学 K1）。"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
        func=mdp.track_ang_vel_z_exp, weight=1.2, params={"command_name": "base_velocity", "std": 0.7071}
    )
    stand_still = RewTerm(func=mdp.stand_still, weight=-0.5, params={"command_name": "base_velocity"})
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces",
               body_names=["Shank_.*", "Hip_.*", ".*hand_.*", "AL.*", "AR.*", "H1", "H2", "Trunk"]),
               "threshold": 1.0},
    )


# ============================================================
# Command curriculum — 学 K1，单阶段全开。
# ============================================================


@configclass
class CurriculumCfg:
    command_ranges = CurrTerm(
        func=mdp.command_range_curriculum,
        params={
            "command_name": "base_velocity",
            "ranges": [
                {
                    "num_steps": 0,
                    "lin_vel_x": (0.0, 2.0),
                    "lin_vel_y": (0.0, 0.0),
                    "ang_vel_z": (-0.3, 0.3),
                },
            ],
        },
    )


@configclass
class TrackingEnvCfg(BaseTrackingEnvCfg):
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        # Match the official deploy-friendly observation convention:
        # policy sees joint positions relative to the action/default offset,
        # and joint velocities are scaled down to reduce OOD amplification.
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel
        self.observations.policy.joint_vel.scale = 0.1
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel
        self.observations.critic.joint_vel.scale = 0.1

    # events: EventCfg = EventCfg()  # 需要时取消注释

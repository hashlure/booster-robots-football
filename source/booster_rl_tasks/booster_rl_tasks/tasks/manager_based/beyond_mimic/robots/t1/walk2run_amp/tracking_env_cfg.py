"""T1 walk2run — body name 适配 + 单阶段全开命令空间（学 K1）。"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from ...k1.walk2run_amp.tracking_env_cfg import (
    RewardsCfg as BaseRewardsCfg,
    TrackingEnvCfg as BaseTrackingEnvCfg,
    EventCfg,
)
import booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp as mdp
import booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.events as bm_events
from isaaclab.managers import EventTermCfg as EvtTerm
from isaaclab.envs.mdp import events as isaac_events


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
        params={"target_height": 0.65}, weight=-1.0)
    keep_balance = RewTerm(func=mdp.stay_alive, weight=0.1)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.7071}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.7071}
    )
    stand_still = RewTerm(func=mdp.stand_still, weight=-3.0, params={"command_name": "base_velocity", "command_threshold": 0.1})
    zero_cmd_penalty = RewTerm(func=mdp.zero_cmd_penalty, weight=-5.0, params={"command_name": "base_velocity", "command_threshold": 0.1})
    # Override base penalties: reduce damping on natural walking motion
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot_.*"]),
        },
    )
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
                    "lin_vel_x": (-0.3, 1.0),
                    "lin_vel_y": (-0.5, 0.5),
                    "ang_vel_z": (-0.5, 0.5),
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
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        # Match the official deploy-friendly observation convention:
        # policy sees joint positions relative to the action/default offset,
        # and joint velocities are scaled down to reduce OOD amplification.
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel
        self.observations.policy.joint_vel.scale = 0.1
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel
        self.observations.critic.joint_vel.scale = 0.1

    # ── Actuator randomization + stand-still push disturbances ──
    @configclass
    class EventCfgAct(EventCfg):
        randomize_actuator_gains = EvtTerm(
            func=isaac_events.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stiffness_distribution_params": (0.85, 1.15),
                "damping_distribution_params": (0.85, 1.15),
                "operation": "scale",
            },
        )
        # Randomize default joint positions (sim2real: deploy pose ≠ train pose)
        randomize_joint_default_pos = EvtTerm(
            func=bm_events.randomize_joint_default_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pos_distribution_params": (-0.02, 0.02),
                "operation": "add",
                "distribution": "uniform",
            },
        )
        reset_robot_joints = EvtTerm(
            func=isaac_events.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.05, 0.05),
                "velocity_range": (0.0, 0.0),
            },
        )
        # More aggressive COM randomization (sim2real: real robot COM ≠ sim)
        base_com = EvtTerm(
            func=bm_events.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
                "com_range": {"x": (-0.03, 0.03), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )
        # Randomize base pitch/roll at reset (trains recovery from tilt, sim2real)
        reset_base = EvtTerm(
            func=isaac_events.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.5, 0.5),
                               "roll": (-0.03, 0.03), "pitch": (-0.17, 0.17)},
                "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.2, 0.2),
                                   "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2)},
            },
        )
        # Multi-directional pushes every 5-8s (not just forward)
        push_robot = EvtTerm(
            func=isaac_events.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 8.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )
    events: EventCfgAct = EventCfgAct()

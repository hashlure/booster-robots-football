"""走向球任务 — 看到球 → 走过去。与 kick_ball_amp 观测空间完全一致，方便后续切换。"""

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

from ..walk2run_amp.tracking_env_cfg import (
    MySceneCfg as BaseMySceneCfg,
    ObservationsCfg as BaseObservationsCfg,
    RewardsCfg as BaseRewardsCfg,
    EventCfg as BaseEventCfg,
    CommandsCfg,
    ActionsCfg,
    TerminationsCfg as BaseTerminationsCfg,
    CurriculumCfg,
    TrackingEnvCfg as BaseTrackingEnvCfg,
)

BALL_RADIUS = 0.11
BALL_MASS = 0.43


# =============================================================================
# 1. 场景 — 与 kick_ball_amp 完全一致
# =============================================================================

@configclass
class MySceneCfg(BaseMySceneCfg):
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.5,
                angular_damping=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, BALL_RADIUS)),
    )


# =============================================================================
# 2. 观测 — 与 kick_ball_amp 完全一致（球位置 3 维）
# =============================================================================

def ball_pos(env, ball_cfg=SceneEntityCfg("ball"), robot_cfg=SceneEntityCfg("robot")):
    from isaaclab.utils.math import quat_apply, quat_conjugate
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    rel_pos = ball.data.root_pos_w - robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    return quat_apply(quat_conjugate(robot_quat), rel_pos)


@configclass
class ObservationsCfg(BaseObservationsCfg):
    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        ball_pos = ObsTerm(func=ball_pos, noise=GaussianNoise(mean=0.0, std=0.02),
                           clip=(-100.0, 100.0), scale=1.0)

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# 3. 奖励 — 核心：靠近球！没有踢球奖励
# =============================================================================

def approach_ball_reward(env, ball_cfg=SceneEntityCfg("ball"), robot_cfg=SceneEntityCfg("robot")):
    """距离越近奖励越大 (1,)。"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    dist = torch.norm(ball.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    return 1.0 / (1.0 + dist)


def reached_ball(env, ball_cfg=SceneEntityCfg("ball"), robot_cfg=SceneEntityCfg("robot")):
    """碰到球的大额奖励 (1,) — 距离小于 0.3m 就给。"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    dist = torch.norm(ball.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    return (dist < 0.3).float() * 10.0


@configclass
class RewardsCfg(BaseRewardsCfg):
    """走向球奖励 — 靠近球的大权重自然引导策略走过去。"""

    ball_approach = RewTerm(func=approach_ball_reward, weight=5.0)
    ball_reach = RewTerm(func=reached_ball, weight=10.0)


# =============================================================================
# 4. 事件 — 与 kick_ball_amp 完全一致（随机放球）
# =============================================================================

def reset_ball_position(env, env_ids, ball_cfg=SceneEntityCfg("ball"), robot_cfg=SceneEntityCfg("robot")):
    """在机器人前方 ±60°、距离 0.6~4m 随机放球。"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    n = len(env_ids)
    dist = torch.rand(n, device=env.device) * 3.4 + 0.6
    angle_local = (torch.rand(n, device=env.device) * 2.0 - 1.0) * (torch.pi / 3.0)  # [-60°, 60°]
    robot_pos = robot.data.root_pos_w[env_ids]
    robot_yaw = robot.data.heading_w[env_ids]
    world_angle = robot_yaw + angle_local
    root_pose = torch.zeros(n, 7, device=env.device)
    root_pose[:, 0] = robot_pos[:, 0] + dist * torch.cos(world_angle)
    root_pose[:, 1] = robot_pos[:, 1] + dist * torch.sin(world_angle)
    root_pose[:, 2] = BALL_RADIUS
    root_pose[:, 3] = 1.0
    ball.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    root_vel = torch.zeros(n, 6, device=env.device)
    ball.write_root_velocity_to_sim(root_vel, env_ids=env_ids)


@configclass
class EventCfg(BaseEventCfg):
    reset_ball = EventTerm(func=reset_ball_position, mode="reset")


# =============================================================================
# 5-6. 终止 / 环境
# =============================================================================

@configclass
class TerminationsCfg(BaseTerminationsCfg):
    pass


@configclass
class TrackingEnvCfg(BaseTrackingEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

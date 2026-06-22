"""踢球任务 — 课程学习：Phase 1 走路靠近 → Phase 2/3 踢球。"""

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

# 继承原始配置
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
import booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp as mdp

# =============================================================================
# 课程学习：根据训练迭代数自动切换阶段
# =============================================================================
# Phase 1 (0-1000):    学走路 + 靠近球，踢球奖励极低
# Phase 2 (1000-2000): 线性过渡，逐渐加入踢球奖励
# Phase 3 (2000-3000): 全力踢球，强调方向和力度
# Phase 4 (3000+):     强化加速度 + 脚弓内侧踢球

_NUM_STEPS_PER_ENV = 24  # 与 ppo_cfg 的 num_steps_per_env 一致


def _estimate_iteration(env) -> float:
    """从环境全局步数估算当前训练迭代数（含 resume 补偿）。"""
    steps = env.common_step_counter if hasattr(env, "common_step_counter") else 0
    offset = getattr(env, "_resume_step_offset", 0)
    return (steps + offset) / (env.num_envs * _NUM_STEPS_PER_ENV)


def _curriculum_scale(env, p1: float, p2: float, p3: float, p4: float | None = None) -> float:
    """四阶段缩放：Phase1(0-1000) → Phase2(1000-2000) → Phase3(2000-3000) → Phase4(3000+)。"""
    if p4 is None:
        p4 = p3
    it = _estimate_iteration(env)
    if it < 1000:
        return p1
    elif it < 2000:
        progress = (it - 1000) / (2000 - 1000)
        return p1 + (p2 - p1) * progress
    elif it < 3000:
        progress = (it - 2000) / (3000 - 2000)
        return p2 + (p3 - p2) * progress
    else:
        return p4


# =============================================================================
# 1. 场景 — 在原有场景上加一个球
# =============================================================================

BALL_RADIUS = 0.11       # 足球半径 (m)
BALL_MASS = 0.43         # 足球质量 (kg)


@configclass
class MySceneCfg(BaseMySceneCfg):
    """继承原始场景，添加足球和球门。"""

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
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.5, 0.0)  # 橙色足球
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5, 0.0, BALL_RADIUS),
        ),
    )

    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/root/booster_amp_lab/booster_assets/models/goal_door.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,       # 球门固定不动
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,       # 球能撞门柱
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(10.0, 0.0, 0.58039),
            rot=(0.5, 0.5, -0.5, -0.5),  # (w,x,y,z) [90,0,-90] 立起+朝向
        ),
    )


# =============================================================================
# 2. 观测 — 球位置 2D + 球门位置 2D
# =============================================================================

# 首触检测：球速超过此阈值即视为已被踢过
KICK_SPEED_THRESHOLD = 0.1  # m/s


def _world_to_robot_2d(env, world_pos, robot_cfg=SceneEntityCfg("robot")):
    """将世界坐标转为机器人坐标系的 2D 位置 (x, y)。"""
    from isaaclab.utils.math import quat_apply, quat_conjugate
    robot = env.scene[robot_cfg.name]
    rel = world_pos - robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    rel_robot = quat_apply(quat_conjugate(robot_quat), rel)
    return rel_robot[:, :2]  # (N, 2)


def ball_pos_2d(env, ball_cfg=SceneEntityCfg("ball"), robot_cfg=SceneEntityCfg("robot")):
    """球在机器人坐标系下的 2D 位置 (x, y)。"""
    ball = env.scene[ball_cfg.name]
    return _world_to_robot_2d(env, ball.data.root_pos_w, robot_cfg)


def goal_pos_2d(env, goal_cfg=SceneEntityCfg("goal"), robot_cfg=SceneEntityCfg("robot")):
    """球门在机器人坐标系下的 2D 位置 (x, y)。"""
    goal = env.scene[goal_cfg.name]
    return _world_to_robot_2d(env, goal.data.root_pos_w, robot_cfg)


def can_approach_flag(env):
    """是否允许靠近球：1=没踢过可以靠近，0=踢过了该站稳。"""
    kicked = getattr(env, "_ball_has_been_kicked",
                     torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return (~kicked).float().unsqueeze(-1)


@configclass
class ObservationsCfg(BaseObservationsCfg):
    """观测 — 无速度指令，球 2D + 球门 2D。AMP 观测继承父类不变。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy 观测：无速度指令，球+球门 2D。"""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=1.0)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025), clip=(-100.0, 100.0), scale=1.0)
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=GaussianNoise(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=GaussianNoise(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        actions = ObsTerm(func=mdp.last_action, noise=GaussianNoise(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        ball_pos = ObsTerm(func=ball_pos_2d, noise=GaussianNoise(mean=0.0, std=0.02), clip=(-100.0, 100.0), scale=1.0)
        goal_pos = ObsTerm(func=goal_pos_2d, noise=GaussianNoise(mean=0.0, std=0.0), clip=(-100.0, 100.0), scale=1.0)
        can_approach = ObsTerm(func=can_approach_flag, clip=(0.0, 1.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Critic 观测：无速度指令，球+球门 2D。"""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        joint_pos = ObsTerm(func=mdp.joint_pos, clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel, clip=(-100.0, 100.0), scale=1.0)
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        ball_pos = ObsTerm(func=ball_pos_2d, clip=(-100.0, 100.0), scale=1.0)
        goal_pos = ObsTerm(func=goal_pos_2d, clip=(-100.0, 100.0), scale=1.0)
        can_approach = ObsTerm(func=can_approach_flag, clip=(0.0, 1.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # AMP 观测继承父类（无速度指令，不需要改）
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


# =============================================================================
# 3. 奖励 — 靠近球 + 踢飞球
# =============================================================================


def _detect_kick(env, ball, robot, ball_speed):
    """判断本次触球是否是真正的踢球（非身体碰撞或球自然滚动）。
    三个条件同时满足才算踢球：
    1. 球速 > KICK_SPEED_THRESHOLD (0.1 m/s)
    2. 至少一只脚离球 < 0.3m（脚在球旁边）
    3. 球在远离机器人（速度方向与球→机器人方向相反，即被踢飞而非弹回）"""
    # 条件1: 球速够快
    is_fast = ball_speed > KICK_SPEED_THRESHOLD

    # 条件2: 脚在球旁边
    feet_ids, _ = robot.find_bodies(name_keys=["left_foot_link", "right_foot_link"], preserve_order=True)
    ball_pos = ball.data.root_pos_w
    dist_lf = torch.norm(robot.data.body_state_w[:, feet_ids[0], :3] - ball_pos, dim=-1)
    dist_rf = torch.norm(robot.data.body_state_w[:, feet_ids[1], :3] - ball_pos, dim=-1)
    foot_near = (dist_lf < 0.3) | (dist_rf < 0.3)

    # 条件3: 球飞离机器人（不是弹回）
    ball_vel_xy = ball.data.root_lin_vel_w[:, :2]
    dir_from_robot = ball_pos[:, :2] - robot.data.root_pos_w[:, :2]
    dir_from_robot = dir_from_robot / (torch.norm(dir_from_robot, dim=-1, keepdim=True) + 1e-6)
    away_from_robot = torch.sum(ball_vel_xy * dir_from_robot, dim=-1) > 0.0

    return is_fast & foot_near & away_from_robot


def approach_ball(env, ball_cfg=SceneEntityCfg("ball"), robot_cfg=SceneEntityCfg("robot"),
                  near_threshold=0.5):
    """势能奖励：距离 > near_threshold 且球静止且未被踢过 时才奖励靠近。
    球一旦被真正踢过（脚近+球速>0.1+球远离），本轮 episode 靠近奖励永久归零。
    课程: Phase1 1.5x → Phase2 线性 → Phase3 1.0x"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    cur_dist = torch.norm(ball.data.root_pos_w - robot.data.root_pos_w, dim=-1)

    if not hasattr(env, "_prev_ball_dist"):
        env._prev_ball_dist = cur_dist.clone()
    if not hasattr(env, "_ball_has_been_kicked"):
        env._ball_has_been_kicked = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    ball_speed = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=-1)
    # 三合一判定真正踢球
    env._ball_has_been_kicked = env._ball_has_been_kicked | _detect_kick(env, ball, robot, ball_speed)

    reward = env._prev_ball_dist - cur_dist
    env._prev_ball_dist = cur_dist.clone()
    # 踢过 或 太近 或 球在动 → 不给
    reward = torch.where(
        ~env._ball_has_been_kicked & (cur_dist > near_threshold) & (ball_speed < 0.01),
        reward, torch.zeros_like(reward))
    return _curriculum_scale(env, 1.5, 1.0, 0.7, 0.5) * reward / env.step_dt


def ball_kick_velocity(env, ball_cfg=SceneEntityCfg("ball"), robot_cfg=SceneEntityCfg("robot")):
    """球飞离机器人的速度分量。
    P1 0.01x(几乎无) → P2 0.5x(有效触球) → P3 1.0x → P4 1.0x"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    ball_vel = ball.data.root_lin_vel_w
    dir_to_ball = ball.data.root_pos_w - robot.data.root_pos_w
    dist = torch.norm(dir_to_ball, dim=-1, keepdim=True)
    dir_to_ball = dir_to_ball / (dist + 1e-6)
    away_speed = torch.sum(ball_vel * dir_to_ball, dim=-1)
    return _curriculum_scale(env, 0.01, 0.5, 1.0, 1.0) * torch.clamp(away_speed, min=0.0)


def ball_kick_acceleration(env, ball_cfg=SceneEntityCfg("ball")):
    """球的瞬时加速度 — 只在球加速时给分（acc与速度同向=踢球，反向=减速不给）。
    P1 0.01x(不踢) → P2 0.5x(支撑脚站稳+摆动脚打到球) → P3 1.0x → P4 1.0x(截断，不暴增)"""
    ball = env.scene[ball_cfg.name]
    lin_acc = ball.data.body_com_acc_w[:, 0, :3]
    lin_vel = ball.data.root_lin_vel_w
    # 加速度在速度方向上的投影 > 0 = 正在加速
    acc_norm = torch.norm(lin_acc, dim=-1)
    vel_norm = torch.norm(lin_vel, dim=-1)
    cos = torch.sum(lin_acc * lin_vel, dim=-1) / (acc_norm * vel_norm + 1e-6)
    speeding_up = cos > 0.0
    return _curriculum_scale(env, 0.01, 0.5, 1.0, 1.0) * speeding_up.float() * torch.clamp(acc_norm, max=5.0)


def ball_toward_goal(env, ball_cfg=SceneEntityCfg("ball"), goal_cfg=SceneEntityCfg("goal")):
    """球速度在球→球门方向上的投影，P3/P4 主导方向奖励。
    P1 0.0x → P2 0.3x(小) → P3 1.5x(方向主导) → P4 1.5x"""
    ball = env.scene[ball_cfg.name]
    goal = env.scene[goal_cfg.name]
    ball_vel_xy = ball.data.root_lin_vel_w[:, :2]
    ball_pos_xy = ball.data.root_pos_w[:, :2]
    goal_xy = goal.data.root_pos_w[:, :2]
    dir_to_goal = goal_xy - ball_pos_xy
    dir_to_goal = dir_to_goal / (torch.norm(dir_to_goal, dim=-1, keepdim=True) + 1e-6)
    toward = torch.sum(ball_vel_xy * dir_to_goal, dim=-1)
    # 时间衰减：球飞越久分越低，避免慢推
    if not hasattr(env, "_ball_moving_acc_time"):
        env._ball_moving_acc_time = torch.zeros(env.num_envs, device=env.device)
    ball_speed = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=-1)
    env._ball_moving_acc_time = torch.where(
        ball_speed > 0.05,
        env._ball_moving_acc_time + env.step_dt,
        torch.zeros_like(env._ball_moving_acc_time),
    )
    decay = torch.exp(-env._ball_moving_acc_time / 1.5)
    return _curriculum_scale(env, 0.0, 0.3, 1.5, 1.5) * torch.clamp(toward, min=0.0) * decay


def body_alignment_to_goal(env, robot_cfg=SceneEntityCfg("robot"), goal_cfg=SceneEntityCfg("goal")):
    """机器人前方方向对准球门。P3/P4 重要。
    P1 0.0x → P2 0.2x(小) → P3 1.0x(方向) → P4 1.0x"""
    from isaaclab.utils.math import quat_apply
    robot = env.scene[robot_cfg.name]
    goal = env.scene[goal_cfg.name]
    n = robot.data.root_pos_w.shape[0]
    goal_pos = goal.data.root_pos_w
    forward_local = torch.tensor([1.0, 0.0, 0.0], device=env.device).unsqueeze(0).expand(n, 3)
    forward_world = quat_apply(robot.data.root_quat_w, forward_local)
    dir_to_goal = goal_pos - robot.data.root_pos_w
    dir_to_goal = dir_to_goal / (torch.norm(dir_to_goal, dim=-1, keepdim=True) + 1e-6)
    alignment = torch.sum(forward_world * dir_to_goal, dim=-1)
    return _curriculum_scale(env, 0.0, 0.2, 1.0, 1.0) * torch.clamp(alignment, min=0.0)


def foot_approach_ball_stationary(env, ball_cfg=SceneEntityCfg("ball"), robot_cfg=SceneEntityCfg("robot"),
                                  speed_threshold=0.01, sigma=0.15):
    """脚靠近静止球。P2 最高(学触球)，P4 很弱(不需要再学靠近)。
    P1 1.0x(基础) → P2 1.5x(重点学触球) → P3 1.0x → P4 0.5x(弱)"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    ball_pos = ball.data.root_pos_w
    ball_speed = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=-1)

    feet_ids, _ = robot.find_bodies(name_keys=["left_foot_link", "right_foot_link"], preserve_order=True)
    lf = robot.data.body_state_w[:, feet_ids[0], :3]
    rf = robot.data.body_state_w[:, feet_ids[1], :3]
    dist_lf = torch.norm(lf - ball_pos, dim=-1)
    dist_rf = torch.norm(rf - ball_pos, dim=-1)
    min_foot_dist = torch.min(dist_lf, dist_rf)

    reward = torch.exp(-min_foot_dist / sigma)
    kicked = getattr(env, "_ball_has_been_kicked", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    reward = torch.where(~kicked & (ball_speed < speed_threshold), reward, torch.zeros_like(reward))
    return _curriculum_scale(env, 1.0, 1.5, 1.0, 0.5) * reward


def stand_still_after_kick(env, robot_cfg=SceneEntityCfg("robot"), sigma=0.2,
                           max_foot_spread=0.4):
    """踢球后保持静止 + 脚距不能太大。
    P1 1.0x → P2 1.0x → P3 2.0x → P4 3.0x"""
    robot = env.scene[robot_cfg.name]
    base_vel_xy = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    kicked = getattr(env, "_ball_has_been_kicked", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))

    # 站立奖励
    still_reward = torch.exp(-base_vel_xy / sigma)

    # 脚距过大扣分
    feet_ids, _ = robot.find_bodies(name_keys=["left_foot_link", "right_foot_link"], preserve_order=True)
    lf = robot.data.body_state_w[:, feet_ids[0], :2]
    rf = robot.data.body_state_w[:, feet_ids[1], :2]
    foot_spread = torch.norm(lf - rf, dim=-1)
    spread_penalty = torch.clamp(foot_spread - max_foot_spread, min=0.0)

    reward = kicked.float() * (still_reward - spread_penalty)
    return _curriculum_scale(env, 1.0, 1.0, 2.0, 3.0) * reward


def rapid_steps_near_ball(env, ball_cfg=SceneEntityCfg("ball"), robot_cfg=SceneEntityCfg("robot"),
                          near_range=1.0):
    """球近时奖励高步频：球距 < near_range 且未踢过时，奖励脚的水平摆动速度。
    课程: P1-P2 0.5x → P3 1.0x → P4 0.5x"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    ball_dist = torch.norm(ball.data.root_pos_w - robot.data.root_pos_w, dim=-1)

    feet_ids, _ = robot.find_bodies(name_keys=["left_foot_link", "right_foot_link"], preserve_order=True)
    lf_vel = torch.norm(robot.data.body_lin_vel_w[:, feet_ids[0], :2], dim=-1)
    rf_vel = torch.norm(robot.data.body_lin_vel_w[:, feet_ids[1], :2], dim=-1)
    foot_speed = (lf_vel + rf_vel) * 0.5

    close = (ball_dist < near_range).float()
    kicked = getattr(env, "_ball_has_been_kicked", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return _curriculum_scale(env, 0.0, 0.0, 1.0, 0.5) * close * (1 - kicked.float()) * foot_speed


def small_stride_near_ball(env, ball_cfg=SceneEntityCfg("ball"), robot_cfg=SceneEntityCfg("robot"),
                           near_range=1.0, max_spread=0.35):
    """球近时惩罚大步幅：左右脚水平距离 > max_spread 扣分，引导小碎步。"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    ball_dist = torch.norm(ball.data.root_pos_w - robot.data.root_pos_w, dim=-1)

    feet_ids, _ = robot.find_bodies(name_keys=["left_foot_link", "right_foot_link"], preserve_order=True)
    lf = robot.data.body_state_w[:, feet_ids[0], :2]
    rf = robot.data.body_state_w[:, feet_ids[1], :2]
    foot_spread = torch.norm(lf - rf, dim=-1)

    close = (ball_dist < near_range).float()
    kicked = getattr(env, "_ball_has_been_kicked", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    overspread = torch.clamp(foot_spread - max_spread, min=0.0)
    return _curriculum_scale(env, 0.0, 0.0, 1.0, 1.0) * close * (1 - kicked.float()) * overspread * -1.0


def instep_kick(env, ball_cfg=SceneEntityCfg("ball"), robot_cfg=SceneEntityCfg("robot"),
                goal_cfg=SceneEntityCfg("goal"), foot_dist=0.25):
    """脚弓内侧推球奖励：1) 球在脚内侧(foot_dist内) 2) 球朝球门飞(>0.5m/s) → 一次性奖励。
    P1-P2 0.0x → P3 0.5x → P4 1.0x"""
    from isaaclab.utils.math import quat_apply, quat_conjugate
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    goal = env.scene[goal_cfg.name]
    if not hasattr(env, "_instep_rewarded"):
        env._instep_rewarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 脚和球在机器人系下的位置
    root_pos = robot.data.root_state_w[:, 0:3]
    root_quat = robot.data.root_state_w[:, 3:7]
    ball_r = quat_apply(quat_conjugate(root_quat), ball.data.root_pos_w - root_pos)
    feet_ids, _ = robot.find_bodies(name_keys=["left_foot_link", "right_foot_link"], preserve_order=True)
    lf_r = quat_apply(quat_conjugate(root_quat), robot.data.body_state_w[:, feet_ids[0], :3] - root_pos)
    rf_r = quat_apply(quat_conjugate(root_quat), robot.data.body_state_w[:, feet_ids[1], :3] - root_pos)

    # 条件1: 球在脚内侧且 < foot_dist
    # 左脚内侧朝 +y → ball_y > foot_y；右脚内侧朝 -y → ball_y < foot_y
    near_lf = torch.norm(ball_r - lf_r, dim=-1) < foot_dist
    near_rf = torch.norm(ball_r - rf_r, dim=-1) < foot_dist
    inside_lf = near_lf & (ball_r[:, 1] > lf_r[:, 1])
    inside_rf = near_rf & (ball_r[:, 1] < rf_r[:, 1])
    on_instep = inside_lf | inside_rf

    # 条件2: 球朝球门飞 > 0.5 m/s
    ball_vel_xy = ball.data.root_lin_vel_w[:, :2]
    ball_pos_xy = ball.data.root_pos_w[:, :2]
    goal_xy = goal.data.root_pos_w[:, :2]
    dir_to_goal = goal_xy - ball_pos_xy
    dir_to_goal = dir_to_goal / (torch.norm(dir_to_goal, dim=-1, keepdim=True) + 1e-6)
    toward_goal = torch.sum(ball_vel_xy * dir_to_goal, dim=-1) > 0.5

    # 首触瞬间一次性奖励
    kick_moment = ~env._instep_rewarded & on_instep & toward_goal
    env._instep_rewarded = env._instep_rewarded | kick_moment
    return _curriculum_scale(env, 0.0, 0.0, 0.5, 1.0) * kick_moment.float()


def foot_clamp_penalty(env, ball_cfg=SceneEntityCfg("ball"), robot_cfg=SceneEntityCfg("robot"),
                       touch_dist=0.20):
    """两重夹球惩罚：
    1. 一脚前一脚后夹球（球x在左右脚x之间）
    2. 左右脚先后碰球（本episode两只脚都碰过球）
    任一条件满足就扣分。"""
    from isaaclab.utils.math import quat_apply, quat_conjugate
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    root_pos = robot.data.root_state_w[:, 0:3]
    root_quat = robot.data.root_state_w[:, 3:7]

    feet_ids, _ = robot.find_bodies(name_keys=["left_foot_link", "right_foot_link"], preserve_order=True)
    lf_w = robot.data.body_state_w[:, feet_ids[0], :3]
    rf_w = robot.data.body_state_w[:, feet_ids[1], :3]
    lf = quat_apply(quat_conjugate(root_quat), lf_w - root_pos)
    rf = quat_apply(quat_conjugate(root_quat), rf_w - root_pos)
    ball_r = quat_apply(quat_conjugate(root_quat), ball.data.root_pos_w - root_pos)

    # 惩罚1: 球x在一只脚前一只脚后
    ball_x, lf_x, rf_x = ball_r[:, 0], lf[:, 0], rf[:, 0]
    ball_between = ((lf_x > ball_x) & (rf_x < ball_x)) | ((lf_x < ball_x) & (rf_x > ball_x))

    # 惩罚2: 左右脚先后碰球（本episode累积）
    if not hasattr(env, "_lf_touched"):
        env._lf_touched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._rf_touched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._lf_touched = env._lf_touched | (torch.norm(lf - ball_r, dim=-1) < touch_dist)
    env._rf_touched = env._rf_touched | (torch.norm(rf - ball_r, dim=-1) < touch_dist)
    both_touched_ever = env._lf_touched & env._rf_touched

    return _curriculum_scale(env, 0.0, 0.0, 1.0, 1.0) * (ball_between | both_touched_ever).float() * -1.0


@configclass
class RewardsCfg(BaseRewardsCfg):
    """踢球奖励 — 继承 walk2run 的正则化项，追加踢球相关，球奖励权重远大于速度跟踪。"""

    # 踢球核心奖励 — 权重大于继承的速度跟踪，自然引导策略走向球并踢飞
    # 1. 势能奖励：每步靠近球 (m/s)
    ball_approach = RewTerm(func=approach_ball, weight=5.0)
    # 2. 球飞离机器人的速度分量 (m/s)，有方向
    ball_kick_vel = RewTerm(func=ball_kick_velocity, weight=3.0)
    # 3. 球瞬时加速度 (m/s²)，捕捉踢球冲击 — 权重最高
    ball_kick_acc = RewTerm(func=ball_kick_acceleration, weight=4.0)
    # 4. 球朝球门方向的速度分量（Phase 4 加时间衰减，鼓励猛射）
    ball_toward_goal = RewTerm(func=ball_toward_goal, weight=5.0)
    # 5. 身体朝向对准球门
    body_alignment_to_goal = RewTerm(func=body_alignment_to_goal, weight=2.0)
    # 6. 脚靠近静止球（球动=无奖励，阻断运球）
    foot_approach_ball_stationary_ = RewTerm(func=foot_approach_ball_stationary, weight=3.0)
    # 7. 踢球后保持静止站立（首触后激活）
    stand_still_after_kick_ = RewTerm(func=stand_still_after_kick, weight=2.0)
    # 8. 球近时快速碎步（球距<1m+未踢过，引导高步频调整站位）
    rapid_steps_near_ball_ = RewTerm(func=rapid_steps_near_ball, weight=2.0)
    # 8b. 球近时惩罚大步幅（脚距>0.5m扣分，引导小碎步）
    small_stride_near_ball_ = RewTerm(func=small_stride_near_ball, weight=5.0)
    # 9. 脚内侧/脚踝触球（侧向发力，P3/P4生效）
    instep_kick_ = RewTerm(func=instep_kick, weight=5.0)
    # 10. 双脚夹球惩罚（一脚前一脚后 或 双脚同时碰球）
    foot_clamp_penalty_ = RewTerm(func=foot_clamp_penalty, weight=3.0)

    # 姿态约束 — 惩罚躯干倾斜（特别是后仰）
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    # 增强z轴速度惩罚，防止跳起
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-5.0)
    # 增强不该碰地部位的惩罚
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-3.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces",
               body_names=[".*_Shank", ".*Hip.*", ".*hand.*", ".*Arm.*", "Head_.*", "Trunk"]),
               "threshold": 1.0},
    )

    # 关闭速度跟踪（踢球任务不需要追踪速度指令）
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=0.0, params={"command_name": "base_velocity", "std": 0.7071}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": 0.7071}
    )
    stand_still = RewTerm(func=mdp.stand_still, weight=0.0, params={"command_name": "base_velocity"})


# =============================================================================
# 4. 事件 — 随机放球位置
# =============================================================================


def reset_ball_position(env, env_ids, ball_cfg=SceneEntityCfg("ball"), robot_cfg=SceneEntityCfg("robot")):
    """在机器人前方 ±60°、距离 0.6~4m 随机放球。同步更新势能奖励 prev_dist。"""
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    n = len(env_ids)
    # 极坐标：距离 0.6~4m，角度在机器人前方 ±60°
    dist = torch.rand(n, device=env.device) * 0.4 + 0.8  # 0.8~1.2m
    angle_local = (torch.rand(n, device=env.device) * 2.0 - 1.0) * (torch.pi / 3.0)  # [-60°, 60°]
    # 转到世界系：机器人位置 + 机器人朝向旋转局部坐标
    robot_pos = robot.data.root_pos_w[env_ids]
    robot_yaw = robot.data.heading_w[env_ids]  # robot heading (rad)
    world_angle = robot_yaw + angle_local
    root_pose = torch.zeros(n, 7, device=env.device)
    root_pose[:, 0] = robot_pos[:, 0] + dist * torch.cos(world_angle)
    root_pose[:, 1] = robot_pos[:, 1] + dist * torch.sin(world_angle)
    root_pose[:, 2] = BALL_RADIUS
    root_pose[:, 3] = 1.0
    ball.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    root_vel = torch.zeros(n, 6, device=env.device)
    ball.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

    # 同步势能奖励的 prev_dist，避免 reset 后第一步出现虚假奖励
    if hasattr(env, "_prev_ball_dist"):
        robot = env.scene[robot_cfg.name]
        new_dist = torch.norm(ball.data.root_pos_w[env_ids] - robot.data.root_pos_w[env_ids], dim=-1)
        env._prev_ball_dist[env_ids] = new_dist

    # 重置球运动计时器 & 加速度累计器 & 首触标记
    if hasattr(env, "_ball_moving_time"):
        env._ball_moving_time[env_ids] = 0.0
    if hasattr(env, "_ball_moving_acc_time"):
        env._ball_moving_acc_time[env_ids] = 0.0
    if hasattr(env, "_last_ball_vel_world"):
        env._last_ball_vel_world[env_ids] = 0.0
    if hasattr(env, "_ball_has_been_kicked"):
        env._ball_has_been_kicked[env_ids] = False
    if hasattr(env, "_instep_rewarded"):
        env._instep_rewarded[env_ids] = False
    if hasattr(env, "_lf_touched"):
        env._lf_touched[env_ids] = False
        env._rf_touched[env_ids] = False


# =============================================================================
# 4.5 终止 — 球运动超时
# =============================================================================

def ball_moving_timeout(env, ball_cfg=SceneEntityCfg("ball"), threshold=0.05, max_time=2.0):
    """球水平速度超过阈值持续 max_time 秒后终止 (N,)。"""
    ball = env.scene[ball_cfg.name]
    ball_speed_xy = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=-1)

    if not hasattr(env, "_ball_moving_time"):
        env._ball_moving_time = torch.zeros(env.num_envs, device=env.device)

    # 累积球运动的时间
    env._ball_moving_time += env.step_dt * (ball_speed_xy > threshold).float()
    return env._ball_moving_time > max_time


def reset_goal_position(env, env_ids, goal_cfg=SceneEntityCfg("goal"), robot_cfg=SceneEntityCfg("robot")):
    """球门放在机器人前方 7~13m（10±3），±15° 范围内。"""
    goal = env.scene[goal_cfg.name]
    robot = env.scene[robot_cfg.name]
    n = len(env_ids)
    robot_pos = robot.data.root_pos_w[env_ids]
    robot_yaw = robot.data.heading_w[env_ids]
    dist = torch.rand(n, device=env.device) * 6.0 + 7.0
    angle = (torch.rand(n, device=env.device) * 2 - 1) * (torch.pi / 12)
    world_angle = robot_yaw + angle
    root_pose = torch.zeros(n, 7, device=env.device)
    root_pose[:, 0] = robot_pos[:, 0] + dist * torch.cos(world_angle)
    root_pose[:, 1] = robot_pos[:, 1] + dist * torch.sin(world_angle)
    root_pose[:, 2] = 0.58039
    root_pose[:, 3] = 0.5    # (w,x,y,z) [90,0,-90]
    root_pose[:, 4] = 0.5
    root_pose[:, 5] = -0.5
    root_pose[:, 6] = -0.5
    goal.write_root_pose_to_sim(root_pose, env_ids=env_ids)


@configclass
class EventCfg(BaseEventCfg):
    """继承原始事件，追加放球+放球门，固定机器人初始位置。"""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    reset_goal = EventTerm(
        func=reset_goal_position,
        mode="reset",
    )

    reset_ball = EventTerm(
        func=reset_ball_position,
        mode="reset",
    )


# =============================================================================
# 5. 终止 — 球被踢远也算一种成功
# =============================================================================

@configclass
class TerminationsCfg(BaseTerminationsCfg):
    """继承原始终止条件 + 球运动超时终止。"""

    ball_moving_timeout = DoneTerm(func=ball_moving_timeout, time_out=True)


# =============================================================================
# 6. 环境配置 — 把上面拼起来
# =============================================================================


@configclass
class TrackingEnvCfg(BaseTrackingEnvCfg):
    """踢球环境 — 只覆盖场景/观测/奖励/事件，其余全部继承。"""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # commands / actions / curriculum 原样继承

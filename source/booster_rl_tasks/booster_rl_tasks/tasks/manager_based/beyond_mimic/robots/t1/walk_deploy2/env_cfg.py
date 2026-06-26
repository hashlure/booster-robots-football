"""T1 deploy — PD 增益和 default_qpos 完全对齐真机 /root/booster_gym/deploy/configs/T1.yaml。"""

from isaaclab.utils import configclass
from booster_rl_tasks.assets.robots.booster import BOOSTER_T1_CFG as ROBOT_CFG
from ..walk2run_amp.tracking_env_cfg import TrackingEnvCfg

# ── 真机 deploy default_qpos（23 维，与 T1.yaml 完全一致）──
DEPLOY_QPOS = {
    ".*_Shoulder_Pitch": 0.2,
    "Left_Shoulder_Roll": -1.35,
    "Right_Shoulder_Roll": 1.35,
    "Left_Elbow_Yaw": -0.5,
    "Right_Elbow_Yaw": 0.5,
    ".*_Hip_Pitch": -0.2,
    ".*_Knee_Pitch": 0.4,
    ".*_Ankle_Pitch": -0.25,
}

# ── 真机 deploy PD 参数（T1.yaml common.stiffness / common.damping）──
DEPLOY_PD = {
    "head":  {"stiffness": 20, "damping": 0.2},
    "arms":  {"stiffness": 20, "damping": 0.5},
    "waist": {"stiffness": 200, "damping": 5},
    "legs":  {"stiffness": 200, "damping": 5},
    "feet":  {"stiffness": 50, "damping": 0},   # parallel mech = 纯P力矩
}


@configclass
class FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 默认关节角 → 真机
        robot_cfg = ROBOT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ROBOT_CFG.init_state.replace(joint_pos=DEPLOY_QPOS),
        )
        # PD 增益 → 真机
        for act_name, act in robot_cfg.actuators.items():
            if act_name in DEPLOY_PD:
                act.stiffness = DEPLOY_PD[act_name]["stiffness"]
                act.damping = DEPLOY_PD[act_name]["damping"]
        self.scene.robot = robot_cfg
        self.actions.joint_pos.use_default_offset = True
        # Phase 1: 低速直走
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

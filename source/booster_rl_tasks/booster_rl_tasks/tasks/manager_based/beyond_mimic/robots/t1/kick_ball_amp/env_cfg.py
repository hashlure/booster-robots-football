"""踢球环境入口 — T1 机器人。"""

from isaaclab.utils import configclass
from booster_rl_tasks.assets.robots.booster import BOOSTER_T1_CFG as ROBOT_CFG, T1_ACTION_SCALE
from .tracking_env_cfg import TrackingEnvCfg


@configclass
class FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = T1_ACTION_SCALE
        self.actions.joint_pos.use_default_offset = True
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

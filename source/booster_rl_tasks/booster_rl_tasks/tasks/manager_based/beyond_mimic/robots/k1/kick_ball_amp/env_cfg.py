"""踢球环境入口 — 继承我们自己的 TrackingEnvCfg，加上 K1 机器人。"""

from isaaclab.utils import configclass
from booster_rl_tasks.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE
from .tracking_env_cfg import TrackingEnvCfg


@configclass
class FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = K1_ACTION_SCALE

"""T1 walking — 五阶段课程控制命令范围。"""

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
        # Phase 1: 低速直走
        # Phase 2: lin=(0.5,1.5), ang=(-0.3,0.3)
        # Phase 4: lin=(1.0,2.5), ang=(-0.5,0.5)
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

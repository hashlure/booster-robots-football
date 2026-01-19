from isaaclab.utils import configclass
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from booster_assets import BOOSTER_ASSETS_DIR
from booster_rl_tasks.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE
from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from .tracking_env_cfg import TrackingEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
import booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
import math

@configclass
class FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.actions.joint_pos.scale = K1_ACTION_SCALE
        self.actions.joint_pos.scale = 0.25



@configclass
class FlatWoStateEstimationEnvCfg(FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        
@configclass
class RoughWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.debug_vis = False        # 设为True可视化地形分布
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(10.0, 10.0),            # 每个地形块尺寸（米）
            border_width=20.0,            # 边界宽度（米）
            num_rows=5,                   # 地形网格行数
            num_cols=10,                  # 地形网格列数
            horizontal_scale=0.1,         # 水平分辨率
            vertical_scale=0.005,         # 垂直分辨率
            slope_threshold=0.75,         # 网格简化阈值
            use_cache=False,              # 每次重新生成地形
            curriculum=False,              # 启用课程学习
            sub_terrains={
                # 80%接近平面的地形（非常平滑）
                "nearly_flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.8,
                    noise_range=(0.0, 0.005),    # 高度波动0-0.5cm（几乎平坦）
                    noise_step=0.005,            # 噪声步长0.5cm
                    border_width=0.25,
                ),
                # 20%随机粗糙地形
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    noise_range=(-0.015, 0.015),    # 高度波动±1.5cm
                    noise_step=0.005,               # 噪声步长0.5cm
                    border_width=0.25,
                ),
            },
        )


@configclass
class StandupEnvCfg(RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.reset_base.params["pose_range"]["x"] = (-0.005, 0.005)
        self.events.reset_base.params["pose_range"]["y"] = (-0.5, 0.5)
        self.events.reset_base.params["pose_range"]["z"] = (-0.005, 0.005)
        self.events.reset_base.params["pose_range"]["roll"] = (-3.14, 3.14)
        self.events.reset_base.params["pose_range"]["pitch"] = (-1.57, -1.57)
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.02, 0.02)
        self.episode_length_s = 3
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.08)
        # self.rewards.base_height = RewTerm(func=mdp.tracking_base_height, params={"target_height": 0.57, "std": 0.5}, weight=1.0)
        self.rewards.head_height = RewTerm(func=mdp.tracking_head_height, params={"target_head_height": 0.8179, "std": 0.3, "asset_cfg": SceneEntityCfg("robot", body_names=["Head_2"])}, weight=2.0)
        self.rewards.stand_still = RewTerm(func= mdp.stand_still, params={"limit_angle": 0.3, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot_.*"])}, weight= -0.5)
        self.rewards.base_height_2 = RewTerm(func=mdp.base_height_l2, params={"target_height": 0.57}, weight=-4.0)
        self.rewards.orientation = RewTerm(func=mdp.flat_orientation_l2, weight = -2)
        self.rewards.hard_stand = RewTerm(func= mdp.get_stand_rew, params={"target_height": 0.8, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot_.*"]), "asset_cfg": SceneEntityCfg("robot", body_names=["Head_2"])}, weight= 1)
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_Shank", "Head_2"]), "threshold": 1.0},
        )
        self.rewards.feet_orientation_l2 = RewTerm(func=mdp.feet_orientation_l2,params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot_.*"])}, weight=-1.0)
        self.curriculum.upward_force_curriculum = CurrTerm(func=mdp.curriculum_force,params={"max_force": 0.0,"asset_cfg": SceneEntityCfg("robot", body_names="Head_2"), "threshold_height": 0.7} )
        self.curriculum.upward_action_scale_curriculum = CurrTerm(func=mdp.curriculum_scale,params={"max_scale":  0.5, "asset_cfg": SceneEntityCfg("robot", body_names="Head_2"),"threshold_height": 0.72} )
        self.events.apply_force = EventTerm(
            func=mdp.apply_curriculum_force,
            mode="interval",
            interval_range_s=(0.0, 0.0),  # 每步执行
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="Head_2"),
            },
        )

@configclass
class LiedownEnvCfg(RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 5
        self.rewards.base_height_2 = RewTerm(func=mdp.base_height_l2, params={"target_height": 0.075}, weight=-2.0)
        self.rewards.liedown_desired_pose = RewTerm(func=mdp.liedown_desired_pose, weight= -1)
        self.rewards.contact_force = RewTerm(func=mdp.contact_force, params={"sensor_cfg": SceneEntityCfg("contact_forces")}, weight= -2)
        self.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-5.0)
        self.rewards.desired_hand_contacts = RewTerm(func= mdp.desired_hand_contacts, params={"height":0.15, "sensor_cfg": SceneEntityCfg("contact_forces", body_names= [".*hand.*"])}, weight= 1.0)
        self.rewards.donot_falling = RewTerm(func=mdp.donot_falling, weight= -2)

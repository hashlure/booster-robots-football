from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from collections.abc import Sequence
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import  CurriculumManager
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def apply_curriculum_force(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    force_value = env.curriculum_manager.get_active_iterable_terms(env_ids)[0][1][0]

    weight = torch.clip(asset.data.root_pos_w[:, 2] / 0.57, 0, 1 )

    if force_value is None or force_value <= 0.0:
        return


    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else 1
    forces = torch.zeros(len(env_ids), num_bodies, 3, device=env.device)
    torques = torch.zeros_like(forces)

    # forces[:, asset_cfg.body_ids[0], 2] = force_value * weight
    # forces[:, asset_cfg.body_ids[1], 2] = force_value * (1- weight)
    forces[:, :, 2] = force_value 
    force_value *= (1- weight)
    # 应用
    asset.set_external_force_and_torque(forces = forces, torques=torques, env_ids=env_ids, body_ids=asset_cfg.body_ids,is_global=True)

def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_actuator_params(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    effort_limit_range: tuple[float, float] = (0.85, 1.15),
    velocity_limit_range: tuple[float, float] = (0.85, 1.15),
    knee_velocity_range: tuple[float, float] = (0.85, 1.15),
    armature_range: tuple[float, float] = (0.80, 1.20),
):
    """Randomly scale actuator parameters (effort/velocity/knee/armature) per-env at reset.

    This mimics manufacturing variance and motor aging for sim2real robustness.

    Args:
        effort_limit_range: min/max scale factor for torque limits.
        velocity_limit_range: min/max scale factor for velocity limits.
        knee_velocity_range: min/max scale factor for knee point velocities.
        armature_range: min/max scale factor for armature.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    device = asset.device

    for act_name, actuator in asset.actuators.items():
        cfg = actuator.cfg
        # These are per-joint dicts or scalars. Handle both.
        for attr_name, scale_range in [
            ("effort_limit_sim", effort_limit_range),
            ("velocity_limit_sim", velocity_limit_range),
            ("knee_point_velocity", knee_velocity_range),
            ("armature", armature_range),
        ]:
            val = getattr(cfg, attr_name, None)
            if val is None:
                continue
            lo, hi = scale_range
            scale = lo + (hi - lo) * torch.rand(len(env_ids), device=device)
            if isinstance(val, dict):
                new_val = {}
                for jn, v in val.items():
                    new_val[jn] = v * scale[0] if isinstance(v, (int, float)) else v * scale
                setattr(cfg, attr_name, new_val)
            elif isinstance(val, (int, float)):
                setattr(cfg, attr_name, val * scale[0])
            # Re-compute stiffness/damping if armature changed
            if attr_name == "armature" and hasattr(cfg, "booster_joint_cfgs"):
                for jn, jc in (cfg.booster_joint_cfgs.items() if isinstance(cfg.booster_joint_cfgs, dict) else {}):
                    jc.armature *= scale[0]
                    if jc.stiffness is not None:
                        w = 2 * 3.1415926535 * jc.natural_freq
                        jc.stiffness = jc.armature * w ** 2
                        jc.damping = 2 * jc.damping_ratio * jc.armature * w


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)

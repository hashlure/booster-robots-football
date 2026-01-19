from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.commands import MotionCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def get_lefthand_pos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """AMP type observations"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elbow_body_ids, _ = asset.find_bodies(name_keys=["left_hand_link", "right_hand_link"], preserve_order=True)
    left_arm_local_vec = torch.tensor([0.0, 0.0, -0.2], device=device).repeat((env.num_envs, 1))
    left_hand_pos = (asset.data.body_state_w[:, elbow_body_ids[0], :3] - asset.data.root_state_w[:, 0:3] + quat_apply(asset.data.body_state_w[:, elbow_body_ids[0], 3:7], left_arm_local_vec))
    left_hand_pos = quat_apply(quat_conjugate(asset.data.root_state_w[:, 3:7]), left_hand_pos)

    return left_hand_pos

def get_righthand_pos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """AMP type observations"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elbow_body_ids, _ = asset.find_bodies(name_keys=["left_hand_link", "right_hand_link"], preserve_order=True)
    right_arm_local_vec = torch.tensor([0.0, 0.0, -0.2], device=device).repeat((env.num_envs, 1))
    right_hand_pos = (asset.data.body_state_w[:, elbow_body_ids[1], :3] - asset.data.root_state_w[:, 0:3] + quat_apply(asset.data.body_state_w[:, elbow_body_ids[1], 3:7], right_arm_local_vec))
    right_hand_pos = quat_apply(quat_conjugate(asset.data.root_state_w[:, 3:7]), right_hand_pos)

    return right_hand_pos

def get_leftfoot_pos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """AMP type observations"""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_body_ids, _ = asset.find_bodies(name_keys=["left_foot_link", "right_foot_link"], preserve_order=True)
    left_foot_pos = (asset.data.body_state_w[:, feet_body_ids[0], :3] - asset.data.root_state_w[:, 0:3])
    left_foot_pos = quat_apply(quat_conjugate(asset.data.root_state_w[:, 3:7]), left_foot_pos)

    return left_foot_pos

def get_rightfoot_pos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """AMP type observations"""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_body_ids, _ = asset.find_bodies(name_keys=["left_foot_link", "right_foot_link"], preserve_order=True)
    right_foot_pos = (asset.data.body_state_w[:, feet_body_ids[1], :3] - asset.data.root_state_w[:, 0:3])
    right_foot_pos = quat_apply(quat_conjugate(asset.data.root_state_w[:, 3:7]), right_foot_pos)

    return right_foot_pos

def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

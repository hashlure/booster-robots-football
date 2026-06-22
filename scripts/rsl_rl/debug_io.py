# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record policy/env I/O for diagnosing locomotion failures."""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import version

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Record detailed RSL-RL policy/env I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--steps", type=int, default=400, help="Number of policy steps to record.")
parser.add_argument("--env_index", type=int, default=0, help="Environment index to record.")
parser.add_argument("--output", type=str, default=None, help="Output .npz path.")
parser.add_argument("--zero_actions", action="store_true", help="Record with zero actions instead of a checkpoint.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import time
import torch

from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner, WMPRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import booster_rl_tasks.tasks  # noqa: F401


def _cpu(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _row(tensor, env_index: int):
    value = _cpu(tensor)
    if value is None:
        return None
    if value.ndim == 0:
        return value
    return value[env_index].copy()


def _make_runner(env, agent_cfg, log_dir):
    runner_class_map = {
        "OnPolicyRunner": OnPolicyRunner,
        "AmpOnPolicyRunner": AmpOnPolicyRunner,
        "WMPRunner": WMPRunner,
    }
    runner_class_name = getattr(agent_cfg, "runner_class_name", "OnPolicyRunner")
    runner_cls = runner_class_map.get(runner_class_name, OnPolicyRunner)
    return runner_cls(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)


def _reward_vector(unwrapped, env_index: int):
    if not hasattr(unwrapped, "reward_manager"):
        return None
    return np.asarray([v[0] for _, v in unwrapped.reward_manager.get_active_iterable_terms(env_index)], dtype=np.float32)


def _contact_forces(unwrapped, env_index: int):
    sensor = getattr(unwrapped.scene, "sensors", {}).get("contact_forces")
    if sensor is None:
        return None
    forces = sensor.data.net_forces_w
    return _row(torch.linalg.norm(forces, dim=-1), env_index)


def _action_term_data(unwrapped, env_index: int):
    if not hasattr(unwrapped, "action_manager"):
        return None, None, []
    raw_parts = []
    processed_parts = []
    joint_names = []
    for name in unwrapped.action_manager.active_terms:
        term = unwrapped.action_manager.get_term(name)
        raw_parts.append(_row(term.raw_actions, env_index))
        processed_parts.append(_row(term.processed_actions, env_index))
        if hasattr(term, "_joint_names"):
            joint_names.extend(list(term._joint_names))
    raw = np.concatenate(raw_parts, axis=0) if raw_parts else None
    processed = np.concatenate(processed_parts, axis=0) if processed_parts else None
    return raw, processed, joint_names


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.env_index < 0 or args_cli.env_index >= env_cfg.scene.num_envs:
        raise ValueError(f"--env_index must be in [0, {env_cfg.scene.num_envs - 1}]")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = None
    log_dir = None
    if not args_cli.zero_actions:
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped

    if args_cli.zero_actions:
        policy = lambda obs: torch.zeros((env.num_envs, env.num_actions), device=unwrapped.device)  # noqa: E731
    else:
        runner = _make_runner(env, agent_cfg, log_dir=None)
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=unwrapped.device)

    output = args_cli.output
    if output is None:
        suffix = "zero" if args_cli.zero_actions else "policy"
        output = os.path.join(log_root_path, f"debug_io_{suffix}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    obs, _ = env.get_observations()
    env_index = args_cli.env_index
    robot = unwrapped.scene["robot"]
    action_term = unwrapped.action_manager.get_term("joint_pos") if hasattr(unwrapped, "action_manager") else None

    records = {
        "obs": [],
        "command": [],
        "policy_action": [],
        "raw_action": [],
        "processed_action": [],
        "joint_pos": [],
        "joint_vel": [],
        "joint_pos_target": [],
        "applied_torque": [],
        "root_lin_vel_b": [],
        "root_ang_vel_b": [],
        "root_pos_w": [],
        "projected_gravity_b": [],
        "reward": [],
        "reward_terms": [],
        "done": [],
        "contact_force_norm": [],
    }

    reward_names = list(unwrapped.reward_manager.active_terms) if hasattr(unwrapped, "reward_manager") else []
    action_joint_names = list(getattr(action_term, "_joint_names", [])) if action_term is not None else []
    robot_joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)

    dt = unwrapped.step_dt
    for step in range(args_cli.steps):
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, rew, dones, extras = env.step(actions)

        raw, processed, names = _action_term_data(unwrapped, env_index)
        if names:
            action_joint_names = names

        records["obs"].append(_row(obs, env_index))
        records["command"].append(_row(unwrapped.command_manager.get_command("base_velocity"), env_index))
        records["policy_action"].append(_row(actions, env_index))
        records["raw_action"].append(raw)
        records["processed_action"].append(processed)
        records["joint_pos"].append(_row(robot.data.joint_pos, env_index))
        records["joint_vel"].append(_row(robot.data.joint_vel, env_index))
        records["joint_pos_target"].append(_row(robot.data.joint_pos_target, env_index))
        records["applied_torque"].append(_row(robot.data.applied_torque, env_index))
        records["root_lin_vel_b"].append(_row(robot.data.root_lin_vel_b, env_index))
        records["root_ang_vel_b"].append(_row(robot.data.root_ang_vel_b, env_index))
        records["root_pos_w"].append(_row(robot.data.root_pos_w, env_index))
        records["projected_gravity_b"].append(_row(robot.data.projected_gravity_b, env_index))
        records["reward"].append(_row(rew, env_index))
        records["reward_terms"].append(_reward_vector(unwrapped, env_index))
        records["done"].append(_row(dones, env_index))
        records["contact_force_norm"].append(_contact_forces(unwrapped, env_index))

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if (step + 1) % 50 == 0:
            action_abs = np.mean(np.abs(records["policy_action"][-50:]))
            target_delta = np.mean(np.abs(np.asarray(records["processed_action"][-50:]) - np.asarray(records["joint_pos"][-50:])))
            base_speed = np.mean(np.linalg.norm(np.asarray(records["root_lin_vel_b"][-50:])[:, :2], axis=-1))
            print(
                f"[DEBUG] step={step + 1:04d} mean|action|={action_abs:.4f} "
                f"mean|target-joint|={target_delta:.4f} mean_base_speed_xy={base_speed:.4f}"
            )

    arrays = {}
    for key, values in records.items():
        if values and values[0] is not None:
            arrays[key] = np.asarray(values)

    metadata = {
        "task": args_cli.task,
        "checkpoint": resume_path or "",
        "zero_actions": str(args_cli.zero_actions),
        "env_index": str(env_index),
        "step_dt": str(dt),
        "sim_dt": str(unwrapped.cfg.sim.dt),
        "decimation": str(unwrapped.cfg.decimation),
        "robot_joint_names": "\n".join(robot_joint_names),
        "action_joint_names": "\n".join(action_joint_names),
        "body_names": "\n".join(body_names),
        "reward_names": "\n".join(reward_names),
    }
    arrays.update({f"meta_{k}": np.asarray(v) for k, v in metadata.items()})
    np.savez_compressed(output, **arrays)

    summary_path = os.path.splitext(output)[0] + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"output: {output}\n")
        for key, value in metadata.items():
            if "\n" not in value:
                f.write(f"{key}: {value}\n")
        f.write("\nrobot_joint_names:\n" + metadata["robot_joint_names"] + "\n")
        f.write("\naction_joint_names:\n" + metadata["action_joint_names"] + "\n")
        f.write("\nreward_names:\n" + metadata["reward_names"] + "\n")

        if "policy_action" in arrays:
            f.write(f"\nmean_abs_policy_action: {np.mean(np.abs(arrays['policy_action'])):.6f}\n")
        if "processed_action" in arrays and "joint_pos" in arrays:
            f.write(f"mean_abs_target_minus_joint: {np.mean(np.abs(arrays['processed_action'] - arrays['joint_pos'])):.6f}\n")
        if "root_lin_vel_b" in arrays:
            f.write(f"mean_base_speed_xy: {np.mean(np.linalg.norm(arrays['root_lin_vel_b'][:, :2], axis=-1)):.6f}\n")
        if "applied_torque" in arrays:
            f.write(f"mean_abs_applied_torque: {np.mean(np.abs(arrays['applied_torque'])):.6f}\n")

    print(f"[INFO] Saved debug data: {output}")
    print(f"[INFO] Saved summary: {summary_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

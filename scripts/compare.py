#!/usr/bin/env python3
"""Compare Isaac Lab vs MuJoCo under fixed velocity commands.

Runs 3 scenarios (stand, walk 0.1, walk 0.2) and logs state/action to CSV.
Usage:
  python scripts/compare.py --checkpoint <path> --num_envs 1 --headless
"""

import argparse, os, csv
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import booster_rl_tasks  # register gym envs
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from rsl_rl.modules import ActorCritic, EmpiricalNormalization

SCENARIOS = [
    ("stand", (0.0, 0.0, 0.0), 500),    # 10s @ 50Hz
    ("walk_0.5", (0.5, 0.0, 0.0), 500),
    ("walk_0.1", (0.1, 0.0, 0.0), 500),
    ("walk_1.0", (1.0, 0.0, 0.0), 500),
]

env_cfg = parse_env_cfg("Booster-Walk2Run-AMP-T1-v0", device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.seed = 42
env = gym.make("Booster-Walk2Run-AMP-T1-v0", cfg=env_cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=None)
robot = env.unwrapped.scene["robot"]
# Get action scale + default qpos for target computation
action_scale = robot.data.default_joint_stiffness  # N/A, use actuators
# Instead use the JointPositionAction offset
act_mgr = env.unwrapped.action_manager
joint_term = act_mgr.get_term("joint_pos")
default_qpos = joint_term._offset[0].clone()  # (23,)
action_scale_vec = joint_term._scale[0].clone()  # (23,)
print(f"Action scale sample: {action_scale_vec[:5]}")
print(f"Default qpos sample: {default_qpos[:5]}")

ckpt = torch.load(args_cli.checkpoint, weights_only=False, map_location="cpu")
ms = ckpt["model_state_dict"]
policy_nn = ActorCritic(
    ms["actor.0.weight"].shape[1], ms["critic.0.weight"].shape[1],
    ms["actor.6.bias"].shape[0],
    actor_hidden_dims=[512,256,128], critic_hidden_dims=[512,256,128],
    activation="elu", noise_std_type="log",
).to(args_cli.device).eval()
policy_nn.load_state_dict(ms)

norm = None
if "obs_norm_state_dict" in ckpt:
    norm = EmpiricalNormalization(
        shape=[ms["actor.0.weight"].shape[1]], until=1e8,
    ).to(args_cli.device).eval()
    norm.load_state_dict(ckpt["obs_norm_state_dict"])

output_dir = "/root/sim2real_data"
os.makedirs(output_dir, exist_ok=True)

for name, cmd, steps in SCENARIOS:
    vx, vy, vyaw = cmd
    print(f"\n[{name}] vx={vx}, vy={vy}, vyaw={vyaw} ...")

    # Fix command
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    cmd_term.cfg.ranges.lin_vel_x = (vx, vx)
    cmd_term.cfg.ranges.lin_vel_y = (vy, vy)
    cmd_term.cfg.ranges.ang_vel_z = (vyaw, vyaw)
    cmd_term.cfg.resampling_time_range = (1e9, 1e9)

    env.reset()
    with torch.no_grad():
        obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    import time
    csv_path = os.path.join(output_dir, f"isaaclab_{name}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    jnames = robot.data.joint_names
    w.writerow(["step","vx","vy","vyaw",
        "pg_x","pg_y","pg_z","angv_x","angv_y","angv_z",
        *[f"q_{n}" for n in jnames],
        *[f"action_{n}" for n in jnames],
        *[f"target_{n}" for n in jnames]])

    for step in range(steps):
        with torch.no_grad():
            if norm is not None:
                obs_n = norm(obs)
            else:
                obs_n = obs
            actions = policy_nn.act_inference(obs_n)
            obs, _, _, _ = env.step(actions)
            if isinstance(obs, tuple):
                obs = obs[0]

        # Compute targets: default_qpos + action * action_scale (BFS order from env)
        targets_isaac = default_qpos + actions[0] * action_scale_vec

        w.writerow([step, vx, vy, vyaw,
            f"{robot.data.projected_gravity_b[0,0].item():.4f}",
            f"{robot.data.projected_gravity_b[0,1].item():.4f}",
            f"{robot.data.projected_gravity_b[0,2].item():.4f}",
            f"{robot.data.root_ang_vel_b[0,0].item():.4f}",
            f"{robot.data.root_ang_vel_b[0,1].item():.4f}",
            f"{robot.data.root_ang_vel_b[0,2].item():.4f}",
            *[f"{robot.data.joint_pos[0,i].item():.4f}" for i in range(robot.num_joints)],
            *[f"{actions[0,i].item():.4f}" for i in range(actions.shape[1])],
            *[f"{targets_isaac[i].item():.4f}" for i in range(len(targets_isaac))]])

    f.close()
    print(f"  -> {csv_path}")

env.close()
print("\nDone. Compare with MuJoCo logs at /tmp/walk2run_mujoco_deploy.log")
simulation_app.close()

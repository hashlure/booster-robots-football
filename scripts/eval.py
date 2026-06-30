#!/usr/bin/env python3
"""Post-training evaluation for walk2run policies.

Runs fixed benchmark scenarios, logs state/action data,
outputs eval_metrics.csv + eval_summary.json.

Usage:
  python scripts/eval.py --task Booster-Walk2Run-AMP-T1-v0 \
      --checkpoint logs/rsl_rl/walk2run_ppo_t1/<run>/model_XXXX.pt \
      --num_envs 256 --headless
"""

import argparse, json, os, time
from collections import defaultdict

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

torch.manual_seed(args_cli.seed); np.random.seed(args_cli.seed)

SCENARIOS = [
    {"name": "stand",         "cmd": (0.0, 0.0, 0.0),  "duration": 20.0},
    {"name": "walk_slow",     "cmd": (0.2, 0.0, 0.0),  "duration": 15.0},
    {"name": "walk_medium",   "cmd": (0.5, 0.0, 0.0),  "duration": 15.0},
    {"name": "walk_fast",     "cmd": (0.8, 0.0, 0.0),  "duration": 15.0},
    {"name": "lateral_left",  "cmd": (0.0, 0.2, 0.0),  "duration": 15.0},
    {"name": "lateral_right", "cmd": (0.0, -0.2, 0.0), "duration": 15.0},
    {"name": "turn_left",     "cmd": (0.0, 0.0, 0.5),  "duration": 15.0},
    {"name": "turn_right",    "cmd": (0.0, 0.0, -0.5), "duration": 15.0},
]

FALL_BASE_HEIGHT = 0.55
FALL_MAX_TILT = 0.7


def is_fallen(pg_z, base_height):
    tilt = float(torch.acos(torch.clamp(-torch.tensor(pg_z), -1.0, 1.0)))
    return base_height < FALL_BASE_HEIGHT or tilt > FALL_MAX_TILT


def compute_metrics(log, dt):
    m = {}
    fell = np.array(log["fell"])
    m["total_steps"] = len(fell)
    m["total_time_s"] = len(fell) * dt
    m["fall_rate"] = float(np.mean(fell))
    if np.any(fell):
        m["survival_time_s"] = float(next(i for i, f in enumerate(fell) if f) * dt)
    else:
        m["survival_time_s"] = len(fell) * dt

    roll = np.array(log["roll"]); pitch = np.array(log["pitch"])
    m["roll_rms_deg"] = float(np.sqrt(np.mean(roll**2)) * 180/np.pi)
    m["pitch_rms_deg"] = float(np.sqrt(np.mean(pitch**2)) * 180/np.pi)
    ang_vel = np.array(log["base_ang_vel"])
    m["base_ang_vel_rms"] = float(np.sqrt(np.mean(ang_vel**2)))

    cmd = np.array(log["commands"]); lin_vel = np.array(log["base_lin_vel"])
    alive = ~fell.astype(bool)
    if np.any(alive):
        m["vx_rmse"] = float(np.sqrt(np.mean((lin_vel[alive,0] - cmd[alive,0])**2)))
        m["vy_rmse"] = float(np.sqrt(np.mean((lin_vel[alive,1] - cmd[alive,1])**2)))
    else:
        m["vx_rmse"] = float("nan"); m["vy_rmse"] = float("nan")

    torque = np.array(log["joint_torque"])
    m["torque_rms"] = float(np.sqrt(np.mean(torque**2)))
    effort_limits = np.array([7.,7.,38.3,38.3,38.3,38.3,38.3,38.3,38.3,38.3,
                              68.,96.,68.,68.,130.,76.,76.,96.,68.,68.,130.,76.,76.])
    sat_ratio = np.abs(torque) / (effort_limits[None,:] + 1e-8)
    m["torque_sat_80_pct"] = float(np.mean(sat_ratio > 0.8) * 100)
    m["torque_sat_95_pct"] = float(np.mean(sat_ratio > 0.95) * 100)

    action = np.array(log["action"])
    if len(action) > 1:
        m["action_rate"] = float(np.mean(np.sum((action[1:]-action[:-1])**2, axis=1)))
    else:
        m["action_rate"] = 0.0

    joint_vel = np.array(log["joint_vel"])
    if len(joint_vel) > 1:
        m["joint_acc_rms"] = float(np.sqrt(np.mean(((joint_vel[1:]-joint_vel[:-1])/dt)**2)))
    else:
        m["joint_acc_rms"] = 0.0

    m["mean_foot_slip"] = 0.0
    m["max_foot_contact_force"] = 0.0
    return m


def run_scenario(env, policy, scenario, dt):
    log = defaultdict(list)
    vx, vy, vyaw = scenario["cmd"]
    max_steps = int(scenario["duration"] / dt)

    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    cmd_term.cfg.ranges.lin_vel_x = (vx, vx)
    cmd_term.cfg.ranges.lin_vel_y = (vy, vy)
    cmd_term.cfg.ranges.ang_vel_z = (vyaw, vyaw)
    cmd_term.cfg.resampling_time_range = (1e9, 1e9)
    if hasattr(cmd_term, "raw_commands") and cmd_term.raw_commands is not None:
        cmd_term.raw_commands[:, 0] = vx
        cmd_term.raw_commands[:, 1] = vy
        cmd_term.raw_commands[:, 2] = vyaw

    with torch.no_grad():
        obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    num_envs = getattr(env, "num_envs", args_cli.num_envs)
    fell = np.zeros(num_envs, dtype=bool)
    robot = env.unwrapped.scene["robot"]

    for step in range(max_steps):
        with torch.no_grad():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if isinstance(obs, tuple):
            obs = obs[0]

        for i in range(num_envs):
            if fell[i]:
                continue
            # Tilt from projected_gravity
            pg = robot.data.projected_gravity_b[i].cpu()
            log["roll"].append(float(torch.atan2(pg[1], -pg[2])))
            log["pitch"].append(float(torch.atan2(-pg[0], torch.sqrt(pg[1]**2+pg[2]**2))))
            log["base_lin_vel"].append(robot.data.root_lin_vel_b[i].cpu().tolist())
            log["base_ang_vel"].append(robot.data.root_ang_vel_b[i].cpu().tolist())
            log["joint_pos"].append(robot.data.joint_pos[i].cpu().tolist())
            log["joint_vel"].append(robot.data.joint_vel[i].cpu().tolist())
            tq = getattr(robot.data, "applied_torque", None)
            log["joint_torque"].append(tq[i].cpu().tolist() if tq is not None else [0.0]*robot.num_joints)
            log["commands"].append([vx, vy, vyaw])
            log["action"].append(actions[i].cpu().tolist())

            base_h = robot.data.root_pos_w[i, 2].item()
            pg_z = robot.data.projected_gravity_b[i, 2].item()
            if is_fallen(pg_z, base_h):
                fell[i] = True
            log["fell"].append(bool(fell[i]))
            log["base_height"].append(base_h)

            # Always break early to log only 1 sample per env per step (avoid 256x log blowup)
            break

        if np.all(fell):
            break

    return dict(log)


def main():
    import booster_rl_tasks  # register gym envs
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from rsl_rl.modules import ActorCritic
    from rsl_rl.modules.normalizer import EmpiricalNormalization

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    dt = env.unwrapped.step_dt

    print(f"[INFO] Loading: {args_cli.checkpoint}")
    ckpt = torch.load(args_cli.checkpoint, weights_only=False, map_location="cpu")
    ms = ckpt["model_state_dict"]
    policy_nn = ActorCritic(
        ms["actor.0.weight"].shape[1], ms["critic.0.weight"].shape[1],
        ms["actor.6.bias"].shape[0],
        actor_hidden_dims=[512,256,128], critic_hidden_dims=[512,256,128],
        activation="elu", noise_std_type="log",
    ).to(args_cli.device).eval()
    policy_nn.load_state_dict(ms)

    normalizer = None
    if "obs_norm_state_dict" in ckpt:
        normalizer = EmpiricalNormalization(
            shape=[ms["actor.0.weight"].shape[1]], until=1.0e8,
        ).to(args_cli.device).eval()
        normalizer.load_state_dict(ckpt["obs_norm_state_dict"])

    def policy(obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        with torch.no_grad():
            if normalizer is not None:
                obs = normalizer(obs)
            return policy_nn.act_inference(obs)

    all_metrics = []
    for s in SCENARIOS:
        print(f"\n[EVAL] {s['name']} (cmd={s['cmd']}) ...")
        with torch.no_grad():
            env.reset()
        log_data = run_scenario(env, policy, s, dt)
        m = compute_metrics(log_data, dt)
        m["scenario"] = s["name"]; m["cmd"] = list(s["cmd"])
        all_metrics.append(m)
        print(f"  fall={m['fall_rate']:.1%} vx_rmse={m['vx_rmse']:.3f} "
              f"sat95={m['torque_sat_95_pct']:.2f}%")

    overall = {}
    for k in all_metrics[0]:
        vals = [m[k] for m in all_metrics if isinstance(m[k], (int,float))]
        overall[k] = float(np.mean(vals)) if vals else all_metrics[0][k]
    overall["scenario"] = "OVERALL"

    out_dir = args_cli.output_dir or os.path.dirname(args_cli.checkpoint)
    os.makedirs(out_dir, exist_ok=True)

    keys = ["scenario","cmd","fall_rate","survival_time_s","vx_rmse","vy_rmse",
            "roll_rms_deg","pitch_rms_deg","base_ang_vel_rms",
            "torque_rms","torque_sat_80_pct","torque_sat_95_pct",
            "action_rate","joint_acc_rms"]
    csv_path = os.path.join(out_dir, "eval_metrics.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(keys)+"\n")
        for m in all_metrics + [overall]:
            f.write(",".join(str(m.get(k,"")) for k in keys)+"\n")
    print(f"\n[EVAL] Saved: {csv_path}")

    json_path = os.path.join(out_dir, "eval_summary.json")
    with open(json_path, "w") as f:
        json.dump({"scenarios": all_metrics, "overall": overall}, f, indent=2)
    print(f"[EVAL] Saved: {json_path}")

    env.close()
    print("\n[EVAL] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()

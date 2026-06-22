#!/usr/bin/env python3
"""convert_act — 双模型自动切换：走向球(walk2run) → 踢球(kick_ball)

根据球距自动切换模型：
- 球距 > switch_distance → walk2run_amp 模型，速度指令指向球
- 球距 ≤ switch_distance → kick_ball_amp 模型，执行踢球

用法:
    python scripts/convert_act.py
    python scripts/convert_act.py --switch_distance 0.8 --num_envs 1
    python scripts/convert_act.py --headless --real-time
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time
from importlib.metadata import version

from isaaclab.app import AppLauncher

# ensure scripts/rsl_rl/ is on path for cli_args import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsl_rl"))

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Dual-model auto-switch: walk to ball → kick ball.")
parser.add_argument("--task", type=str, default="Booster-KickBall-AMP-v0",
                    help="Isaac Lab task (default: Booster-KickBall-AMP-v0)")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="RL agent config entry point")
parser.add_argument("--walk2run_checkpoint", type=str, default=None,
                    help="Path to walk2run model checkpoint. Auto-discover if not set.")
parser.add_argument("--kickball_checkpoint", type=str, default=None,
                    help="Path to kick_ball model checkpoint. Auto-discover if not set.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of environments (default: 1)")
parser.add_argument("--switch_distance", type=float, default=0.6,
                    help="Ball distance threshold for switching (m, default: 0.6)")
parser.add_argument("--max_vel_x", type=float, default=1.0,
                    help="Max forward velocity for walk2run (m/s, default: 1.0)")
parser.add_argument("--start_model", type=str, default="walk2run",
                    choices=["walk2run", "kick_ball"],
                    help="Starting model (default: walk2run)")
parser.add_argument("--real-time", action="store_true", default=True,
                    help="Run in real-time with sleep delays")
# append RSL-RL cli args
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.modules import ActorCritic, EmpiricalNormalization

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs import multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_rl_tasks.tasks  # noqa: F401


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

POLICY_ARCH = {
    "actor_hidden_dims": [512, 256, 128],
    "critic_hidden_dims": [512, 256, 128],
    "activation": "elu",
}


def load_amp_policy(checkpoint_path, num_actor_obs, num_critic_obs, num_actions, device):
    """Load an AMP model checkpoint directly (no AmpOnPolicyRunner needed).

    Args:
        checkpoint_path: Path to .pt checkpoint.
        num_actor_obs: Actor observation dimension (75 for walk2run, 77 for kick_ball).
        num_critic_obs: Critic observation dimension (78 for walk2run, 80 for kick_ball).
        num_actions: Action dimension (22 for K1).
        device: torch device.

    Returns:
        Callable: policy_fn(obs) -> actions, with normalizer baked in.
    """
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)

    policy = ActorCritic(
        num_actor_obs, num_critic_obs, num_actions,
        actor_hidden_dims=POLICY_ARCH["actor_hidden_dims"],
        critic_hidden_dims=POLICY_ARCH["critic_hidden_dims"],
        activation=POLICY_ARCH["activation"],
        noise_std_type="log",
    ).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    normalizer = EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8).to(device)
    if "obs_norm_state_dict" in ckpt:
        normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
    else:
        print("[WARN] No obs_norm_state_dict in checkpoint — using identity normalizer")
    normalizer.eval()

    def policy_fn(obs):
        with torch.inference_mode():
            return policy.act_inference(normalizer(obs))

    print(f"  actor input dim: {num_actor_obs}, critic input dim: {num_critic_obs}, action dim: {num_actions}")
    return policy_fn


# ---------------------------------------------------------------------------
# Observation conversion
# ---------------------------------------------------------------------------

def convert_kickball_to_walk2run(obs_77, vel_cmd):
    """Convert kick_ball env observation (77 dims) to walk2run format (75 dims).

    kick_ball obs layout:
        [0:3]   base_ang_vel      [28:50] joint_vel
        [3:6]   projected_gravity [50:72] last_action
        [6:28]  joint_pos         [72:74] ball_pos_2d
                                   [74:76] goal_pos_2d
                                   [76:77] can_approach

    walk2run obs layout:
        [0:3]   base_ang_vel      [31:53] joint_vel
        [3:6]   projected_gravity [53:75] last_action
        [6:9]   velocity_commands  ← supplied by caller

    Args:
        obs_77:  (N, 77) tensor from kick_ball env.
        vel_cmd: (N, 3)  tensor [lin_vel_x, lin_vel_y, ang_vel_z].

    Returns:
        (N, 75) tensor for walk2run model.
    """
    return torch.cat([
        obs_77[:, 0:6],      # base_ang_vel(3) + projected_gravity(3)
        vel_cmd,              # velocity_commands(3) — 指向球
        obs_77[:, 6:28],     # joint_pos(22)
        obs_77[:, 28:50],    # joint_vel(22)
        obs_77[:, 50:72],    # last_action(22)
    ], dim=1)


# ---------------------------------------------------------------------------
# Velocity command: steer toward ball
# ---------------------------------------------------------------------------

def compute_velocity_toward_ball(ball_pos_robot, max_vel_x=1.0):
    """Compute walk2run velocity command that steers the robot toward the ball.

    Args:
        ball_pos_robot: (N, 2) ball position in robot frame [x, y].
        max_vel_x: Max forward speed (m/s).

    Returns:
        (N, 3) tensor [lin_vel_x, lin_vel_y, ang_vel_z].
    """
    dist = torch.norm(ball_pos_robot, dim=-1)
    angle = torch.atan2(ball_pos_robot[:, 1], ball_pos_robot[:, 0])

    # Forward speed: proportional to distance, capped
    lin_vel_x = torch.clamp(dist * 0.5, max=max_vel_x)
    lin_vel_y = torch.zeros_like(lin_vel_x)

    # Turn rate: proportional to angle, capped at ±0.5 rad/s
    ang_vel_z = torch.clamp(angle * 1.5, min=-0.5, max=0.5)

    return torch.stack([lin_vel_x, lin_vel_y, ang_vel_z], dim=-1)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoint(log_subdir):
    """Find the most recent model checkpoint under logs/rsl_rl/<log_subdir>/."""
    log_root = os.path.join("logs", "rsl_rl", log_subdir)
    if not os.path.isdir(log_root):
        return None

    # iterate all run dirs, pick newest checkpoint
    best = None
    best_iter = -1
    for run in os.listdir(log_root):
        run_dir = os.path.join(log_root, run)
        if not os.path.isdir(run_dir):
            continue
        for f in os.listdir(run_dir):
            if f.startswith("model_") and f.endswith(".pt"):
                try:
                    it = int(f.replace("model_", "").replace(".pt", ""))
                except ValueError:
                    continue
                if it > best_iter:
                    best_iter = it
                    best = os.path.join(run_dir, f)
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlOnPolicyRunnerCfg):
    """Dual-model auto-switch: walk2run → approach ball → kick_ball."""

    # --- resolve checkpoints ---
    walk2run_ckpt = args_cli.walk2run_checkpoint
    if walk2run_ckpt is None:
        walk2run_ckpt = find_checkpoint("walk2run_amp")
    if walk2run_ckpt is None:
        raise FileNotFoundError(
            "No walk2run checkpoint found. Use --walk2run_checkpoint to specify one."
        )

    kickball_ckpt = args_cli.kickball_checkpoint
    if kickball_ckpt is None:
        kickball_ckpt = find_checkpoint("kick_ball_amp")
    if kickball_ckpt is None:
        raise FileNotFoundError(
            "No kick_ball checkpoint found. Use --kickball_checkpoint to specify one."
        )

    print(f"[INFO] walk2run checkpoint : {walk2run_ckpt}")
    print(f"[INFO] kick_ball checkpoint: {kickball_ckpt}")

    # --- env setup ---
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # override ball initial position: 4m in front of robot
    BALL_RADIUS = 0.11
    env_cfg.scene.ball.init_state.pos = (4.0, 0.0, BALL_RADIUS)
    print(f"[INFO] Ball init position set to: {env_cfg.scene.ball.init_state.pos}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    device = args_cli.device
    dt = env.unwrapped.step_dt
    num_envs = env.unwrapped.num_envs
    print(f"[INFO] env: {num_envs} env(s), step_dt={dt:.4f}s ({1.0/dt:.0f} Hz)")

    # --- load models ---
    walk2run_fn = load_amp_policy(walk2run_ckpt, num_actor_obs=75, num_critic_obs=78, num_actions=22, device=device)
    kickball_fn = load_amp_policy(kickball_ckpt, num_actor_obs=77, num_critic_obs=80, num_actions=22, device=device)

    # --- state ---
    active_model = args_cli.start_model
    switch_distance = args_cli.switch_distance
    slowdown_buffer = 0.4            # distance before switch_distance where velocity ramps to 0
    max_vel_x = args_cli.max_vel_x
    switch_time = -999.0             # last switch timestamp (for debounce)
    debounce = 1.0                   # min seconds between switches

    vel_cmd = torch.zeros(num_envs, 3, device=device)

    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = obs

    step_count = 0
    print(f"[INFO] Starting with model: {active_model}")
    print(f"[INFO] Switch distance: {switch_distance} m  |  Slowdown buffer: {slowdown_buffer} m  |  Max vel_x: {max_vel_x} m/s")
    print(f"[INFO] Press Ctrl+C to stop.\n")

    # --- simulation loop ---
    while simulation_app.is_running():
        loop_start = time.time()

        # ball distance from robot (ball in robot frame from obs)
        ball_pos_robot = obs[:, 72:74]        # (N, 2) in robot frame
        ball_dist = torch.norm(ball_pos_robot, dim=-1)  # (N,)
        d = ball_dist.mean().item()

        # raw velocity command toward ball
        vel_raw = compute_velocity_toward_ball(ball_pos_robot, max_vel_x)

        # --- velocity scaling: ramp to 0 as we approach switch_distance ---
        #  d ≥ switch_distance + slowdown_buffer  →  speed_scale = 1.0  (full speed)
        #  switch_distance < d < ... + buffer      →  speed_scale = 0→1 (linear ramp)
        #  d ≤ switch_distance                     →  speed_scale = 0.0  (stopped)
        if d >= switch_distance + slowdown_buffer:
            speed_scale = 1.0
        elif d > switch_distance:
            speed_scale = (d - switch_distance) / slowdown_buffer
        else:
            speed_scale = 0.0

        vel_target = vel_raw * speed_scale
        # EMA smoothing
        vel_cmd = vel_cmd * 0.9 + vel_target * 0.1

        # --- model switching ---
        now = time.time()
        if now - switch_time > debounce:
            # switch to walk2run when ball is far AND we're not already walking
            if d > switch_distance and active_model != "walk2run":
                active_model = "walk2run"
                switch_time = now
                print(f"\n[SWITCH @ step {step_count}] → walk2run  (ball_dist={d:.3f}m)")
            # switch to kick_ball when ball is close AND we've stopped
            elif d <= switch_distance and speed_scale == 0.0 and active_model != "kick_ball":
                active_model = "kick_ball"
                switch_time = now
                print(f"\n[SWITCH @ step {step_count}] → kick_ball  (ball_dist={d:.3f}m)")

        # --- inference ---
        with torch.inference_mode():
            if active_model == "walk2run":
                obs_75 = convert_kickball_to_walk2run(obs, vel_cmd)
                actions = walk2run_fn(obs_75)
            else:
                actions = kickball_fn(obs)

        # --- step env ---
        obs, _, _, _ = env.step(actions)
        if isinstance(obs, tuple):
            obs = obs[0]

        step_count += 1

        # --- status print ---
        if step_count % 200 == 0:
            print(f"[{step_count:6d}] model={active_model:>10s}  "
                  f"ball_dist={d:.3f}m  speed={speed_scale:.2f}  "
                  f"vel=({vel_cmd[0,0]:.2f}, {vel_cmd[0,1]:.2f}, {vel_cmd[0,2]:.2f})")

        # --- real-time sleep ---
        if args_cli.real_time:
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    env.close()
    print(f"\n[DONE] {step_count} steps total.")


if __name__ == "__main__":
    main()
    simulation_app.close()

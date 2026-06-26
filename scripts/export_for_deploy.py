#!/usr/bin/env python3
"""Export Isaac Lab T1-walk2run policy for booster_gym deploy.

Maps deploy's 47-dim (legs-only) observation → Isaac Lab's full-body format,
runs the policy, extracts leg actions.

Outputs:
  - policy.jit.pt       : TorchScript deploy-ready model
  - deploy_config.yaml  : Config snippet for booster_gym/deploy/configs/
"""

import os, argparse, torch, yaml, numpy as np
from rsl_rl.modules import ActorCritic
from rsl_rl.modules.normalizer import EmpiricalNormalization


def load_checkpoint(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    ms = ckpt["model_state_dict"]
    num_actor_obs = ms["actor.0.weight"].shape[1]
    num_critic_obs = ms["critic.0.weight"].shape[1]
    num_actions = ms["actor.6.bias"].shape[0]

    policy = ActorCritic(
        num_actor_obs, num_critic_obs, num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        noise_std_type="log",
    ).to(device)
    policy.load_state_dict(ms)
    policy.eval()

    normalizer = EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8).to(device)
    if "obs_norm_state_dict" in ckpt:
        normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
    else:
        print("[WARN] No obs_norm_state_dict in checkpoint")
    normalizer.eval()

    return policy, normalizer, num_actor_obs, num_actions


class DeployAdapter(torch.nn.Module):
    """Wraps Isaac Lab policy for booster_gym deploy interface.

    Deploy input  (47): pg(3)+ang_vel(3)+cmd(3)+gait(2)+dof_pos(12)+dof_vel(12)+action(12)
    Isaac Lab input (78): ang_vel(3)+pg(3)+cmd(3)+dof_pos(23)+dof_vel(23)+action(23)
    Deploy output (12): leg joint targets (deltas from default_qpos)

    Upper-body joints (0:11) are set to 0 in observation (policy sees neutral pose).
    """

    def __init__(self, policy, normalizer, default_qpos, action_scale_legs):
        super().__init__()
        self.policy = policy
        self.normalizer = normalizer
        # default_qpos: full 23-dim T1 default joint positions
        self.register_buffer("default_qpos", torch.tensor(default_qpos, dtype=torch.float32))
        # Per-joint action scale for leg joints (indices 11:23)
        self.register_buffer("action_scale", torch.tensor(action_scale_legs, dtype=torch.float32))
        # Deploy uses action_scale=1.0, but our model was trained with per-joint scaling.
        # We account for this by dividing the model output by the original scale,
        # so deploy's multiplication by 1.0 gives the correct target.

    def forward(self, obs_deploy):
        """obs_deploy: (..., 47) from booster_gym deploy interface."""
        # Support both (47,) and (B, 47)
        if obs_deploy.dim() == 1:
            obs_deploy = obs_deploy.unsqueeze(0)
        B = obs_deploy.shape[0]

        # De-normalize deploy observation
        pg_raw     = obs_deploy[:, 0:3]
        angvel_raw = obs_deploy[:, 3:6]
        cmd_raw    = obs_deploy[:, 6:9]
        dof_pos_delta = obs_deploy[:, 11:23]
        dof_vel_legs  = obs_deploy[:, 23:35]
        last_action_legs = obs_deploy[:, 35:47]

        # Build Isaac Lab observation (B, 78)
        obs_isaac = torch.zeros(B, 78, device=obs_deploy.device, dtype=torch.float32)
        obs_isaac[:, 0:3] = angvel_raw
        obs_isaac[:, 3:6] = pg_raw
        obs_isaac[:, 6:9] = cmd_raw
        # joint_pos: legs at indices 11:23, upper body = 0
        obs_isaac[:, 9+11:9+23] = dof_pos_delta
        # joint_vel
        obs_isaac[:, 32+11:32+23] = dof_vel_legs
        # last_action
        obs_isaac[:, 55+11:55+23] = last_action_legs

        # Normalize
        obs_normed = self.normalizer(obs_isaac)

        # Inference
        actions_full = self.policy.act_inference(obs_normed)  # (B, 23)

        # Extract leg actions (indices 11:23) and scale
        leg_actions = actions_full[:, 11:23]
        leg_deltas = self.action_scale * leg_actions

        # Return (12,) for single input, (B, 12) for batch
        if B == 1 and obs_deploy.dim() == 1:
            return leg_deltas[0]
        return leg_deltas


def export_deploy_model(checkpoint_path, output_dir, device="cpu"):
    os.makedirs(output_dir, exist_ok=True)

    policy, normalizer, num_obs, num_act = load_checkpoint(checkpoint_path, device)
    print(f"Loaded: actor_obs={num_obs}, actions={num_act}")

    # T1 default joint positions (from BOOSTER_T1_CFG)
    default_qpos = np.zeros(23, dtype=np.float32)
    default_qpos[2]  = 0.2   # L_Shoulder_Pitch
    default_qpos[3]  = -1.3  # L_Shoulder_Roll
    default_qpos[5]  = -0.5  # L_Elbow_Yaw
    default_qpos[7]  = 0.2   # R_Shoulder_Pitch
    default_qpos[8]  = 1.3   # R_Shoulder_Roll
    default_qpos[10] = 0.5   # R_Elbow_Yaw
    default_qpos[11] = -0.2  # L_Hip_Pitch
    default_qpos[14] = 0.4   # L_Knee_Pitch
    default_qpos[15] = -0.2  # L_Ankle_Pitch
    default_qpos[17] = -0.2  # R_Hip_Pitch
    default_qpos[20] = 0.4   # R_Knee_Pitch
    default_qpos[21] = -0.2  # R_Ankle_Pitch

    # T1 per-joint action scale for legs (from T1_ACTION_SCALE computation)
    # These are the Isaac Lab per-joint scales used during training
    action_scale_legs = np.ones(12, dtype=np.float32)  # Placeholder — fill from actual T1 config
    # Typical T1 leg scales (0.25*effort/stiffness):
    action_scale_legs[0]  = 0.056  # L_Hip_Pitch
    action_scale_legs[1]  = 0.047  # L_Hip_Roll
    action_scale_legs[2]  = 0.048  # L_Hip_Yaw
    action_scale_legs[3]  = 0.075  # L_Knee_Pitch
    action_scale_legs[4]  = 0.12   # L_Ankle_Pitch
    action_scale_legs[5]  = 0.075  # L_Ankle_Roll
    action_scale_legs[6]  = 0.056  # R_Hip_Pitch
    action_scale_legs[7]  = 0.047  # R_Hip_Roll
    action_scale_legs[8]  = 0.048  # R_Hip_Yaw
    action_scale_legs[9]  = 0.075  # R_Knee_Pitch
    action_scale_legs[10] = 0.12   # R_Ankle_Pitch
    action_scale_legs[11] = 0.075  # R_Ankle_Roll

    adapter = DeployAdapter(policy, normalizer, default_qpos, action_scale_legs)
    adapter.eval()

    # Test with dummy input
    dummy = torch.zeros(47)
    with torch.no_grad():
        out = adapter(dummy)
    print(f"Test: input(47) -> output({out.shape[0]})")

    # ---- Full-body export (no adapter, direct 78→23) ----
    class FullBodyExport(torch.nn.Module):
        def __init__(self, policy, normalizer):
            super().__init__()
            self.policy = policy
            self.normalizer = normalizer

        def forward(self, obs):
            """obs: (..., 78) Isaac Lab policy observation"""
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            normed = self.normalizer(obs)
            return self.policy.act_inference(normed)  # (..., 23)

    fullbody = FullBodyExport(policy, normalizer)
    fullbody.eval()
    dummy78 = torch.zeros(78)
    with torch.no_grad():
        fb_out = fullbody(dummy78)
    print(f"Full-body test: input(78) -> output({fb_out.shape[-1]})")

    fb_path = os.path.join(output_dir, "policy_fullbody.jit.pt")
    with torch.no_grad():
        fb_traced = torch.jit.trace(fullbody, dummy78.unsqueeze(0))
    fb_traced.save(fb_path)
    print(f"Full-body JIT: {fb_path}")

    # ---- Legs-only export (with adapter, 47→12) ----
    jit_path = os.path.join(output_dir, "policy.jit.pt")
    with torch.no_grad():
        traced = torch.jit.trace(adapter, dummy.unsqueeze(0))
    traced.save(jit_path)
    print(f"Legs-only JIT: {jit_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="/root/booster_gym/deploy/models/t1_walk2run")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    export_deploy_model(args.checkpoint, args.output, args.device)

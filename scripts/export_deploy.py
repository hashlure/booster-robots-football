#!/usr/bin/env python3
"""Export Isaac Lab deploy-trained policy to TorchScript for booster_gym deploy.

Usage:
  python scripts/export_deploy.py --checkpoint logs/rsl_rl/walk_deploy_t1/.../model_4000.pt
"""

import argparse, torch
from rsl_rl.modules import ActorCritic

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", default="/root/booster_gym/deploy/models/t1_walk_deploy/policy.jit.pt")
args = parser.parse_args()

ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
ms = ckpt["model_state_dict"]
num_actor_obs = ms["actor.0.weight"].shape[1]
num_critic_obs = ms["critic.0.weight"].shape[1]
num_actions = ms["actor.6.bias"].shape[0]

policy = ActorCritic(num_actor_obs, num_critic_obs, num_actions,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu", noise_std_type="log").eval()
policy.load_state_dict(ms)

# Wrap for tracing (forward() is empty, act_inference is the real forward)
class PolicyWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
    def forward(self, obs):
        return self.policy.act_inference(obs)

wrapped = PolicyWrapper(policy).eval()

# Test
dummy = torch.zeros(1, num_actor_obs)
with torch.no_grad():
    out = wrapped(dummy)
print(f"Test: input({num_actor_obs}) → output({out.shape[1]})")

# Export
traced = torch.jit.trace(wrapped, dummy)
traced.save(args.output)
print(f"Saved: {args.output}")

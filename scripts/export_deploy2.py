#!/usr/bin/env python3
"""Export walk_deploy2 model for deploy_fullbody.py."""

import torch, argparse
from rsl_rl.modules import ActorCritic
from rsl_rl.modules.normalizer import EmpiricalNormalization

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", default="/root/booster_gym/deploy/models/t1_walk2run/policy.jit.pt")
args = parser.parse_args()

ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
ms = ckpt["model_state_dict"]
num_actor_obs = ms["actor.0.weight"].shape[1]
num_critic_obs = ms["critic.0.weight"].shape[1]
num_actions = ms["actor.6.bias"].shape[0]

policy = ActorCritic(num_actor_obs, num_critic_obs, num_actions,
    actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
    activation="elu", noise_std_type="log").eval()
policy.load_state_dict(ms)

normalizer = None
if "obs_norm_state_dict" in ckpt:
    normalizer = EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8)
    normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
    normalizer.eval()

class PolicyWithNorm(torch.nn.Module):
    def __init__(self, p, n):
        super().__init__()
        self.policy = p
        self.normalizer = n
    def forward(self, obs):
        if self.normalizer is not None:
            obs = self.normalizer(obs)
        return self.policy.act_inference(obs)

wrapped = PolicyWithNorm(policy, normalizer).eval()
dummy = torch.zeros(1, num_actor_obs)
with torch.no_grad():
    out = wrapped(dummy)
print(f"Exported: obs={num_actor_obs}, act={out.shape[1]}, normalizer={normalizer is not None}")

traced = torch.jit.trace(wrapped, dummy)
traced.save(args.output)
print(f"Saved: {args.output}")

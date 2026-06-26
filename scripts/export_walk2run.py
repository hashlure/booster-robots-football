#!/usr/bin/env python3
"""Export walk2run model (78-dim obs + empirical normalizer) for deploy."""

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

normalizer = EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8)
if "obs_norm_state_dict" in ckpt:
    normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
    print(f"Normalizer loaded (mean first 3): {normalizer.mean[:3]}")
else:
    print("WARN: no normalizer in checkpoint")
normalizer.eval()

class PolicyWithNorm(torch.nn.Module):
    def __init__(self, policy, normalizer):
        super().__init__()
        self.policy = policy
        self.normalizer = normalizer
    def forward(self, obs):
        return self.policy.act_inference(self.normalizer(obs))

wrapped = PolicyWithNorm(policy, normalizer).eval()

dummy = torch.zeros(1, num_actor_obs)
with torch.no_grad():
    out = wrapped(dummy)
print(f"Test: input({num_actor_obs}) -> output({out.shape[1]})")

traced = torch.jit.trace(wrapped, dummy)
traced.save(args.output)
print(f"Saved: {args.output}")

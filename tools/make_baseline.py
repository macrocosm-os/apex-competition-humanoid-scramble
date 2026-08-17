"""Build baseline/baseline.onnx from Unitree's stock G1 walking policy.

The baseline is not trained for this course — it is Unitree's flat-ground `motion.pt` wrapped so
it speaks the competition's interface. That is deliberate: `defaults.baseline_raw_score` only has
to be a real, reproducible score that a submission must beat, and shipping the stock walker means
the number is honestly "what a competent off-the-shelf locomotion policy does here".

The wrapper does three things:
  1. slices the competition's 104-d observation into the 47-d vector motion.pt was trained on,
  2. synthesises the velocity command motion.pt expects, closing a heading-hold loop on the
     course centreline — motion.pt tracks BODY-FRAME velocity with no heading feedback, so
     without this it drifts ~0.26 m sideways per metre and leaves the track before any obstacle,
  3. ignores perception entirely. It cannot see the course. That is the ceiling being offered.

motion.pt is an LSTM, and its TorchScript build hides the recurrent state inside the module
(mutating buffers in `forward`), which neither ONNX exporter will trace. So the weights are
lifted into a plain nn.Module that threads state through the competition's `state_in`/`state_out`
tensors. `--check` asserts the rebuild is numerically identical to the original.

    python tools/make_baseline.py --urlg <unitree_rl_gym checkout>

Needs torch. See tools/preview_v3.py for the sparse-checkout command.
"""

from __future__ import annotations

import argparse
import pathlib

import torch
from torch import nn

from env.sim import ACT_DIM, OBS_DIM, STATE_DIM

# Indices into the competition observation (env/sim.py).
I_GRAV, I_ANGVEL, I_QPOS, I_QVEL, I_ACT, I_PHASE, I_YAW, I_Y = 0, 3, 9, 21, 33, 45, 47, 49

CMD_SCALE = torch.tensor([2.0, 2.0, 0.25])
FORWARD_SPEED = 0.8
H = 64  # motion.pt LSTM width


class BaselinePolicy(nn.Module):
    """Stock G1 walker, competition interface, explicit recurrent state."""

    def __init__(self, src: torch.jit.ScriptModule):
        super().__init__()
        self.lstm = nn.LSTM(47, H, batch_first=False)
        self.actor = nn.Sequential(nn.Linear(H, 32), nn.ELU(), nn.Linear(32, ACT_DIM))
        sd = dict(src.named_parameters())
        with torch.no_grad():
            for k in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
                getattr(self.lstm, k).copy_(sd[f"memory.{k}"])
            self.actor[0].weight.copy_(sd["actor.0.weight"])
            self.actor[0].bias.copy_(sd["actor.0.bias"])
            self.actor[2].weight.copy_(sd["actor.2.weight"])
            self.actor[2].bias.copy_(sd["actor.2.bias"])
        self.register_buffer("cmd_scale", CMD_SCALE)

    def forward(self, obs: torch.Tensor, state_in: torch.Tensor):
        # Heading hold on the centreline. sin(yaw) stands in for yaw itself: the loop keeps the
        # error small, where the two agree to well under a degree, and it avoids atan2 in ONNX.
        sin_yaw = obs[:, I_YAW:I_YAW + 1]
        y = obs[:, I_Y:I_Y + 1]
        cmd = torch.cat([
            torch.full_like(y, FORWARD_SPEED),
            torch.clamp(-0.5 * y, -0.3, 0.3),
            torch.clamp(-1.5 * sin_yaw - 0.8 * y, -0.6, 0.6),
        ], dim=1) * self.cmd_scale

        u = torch.cat([
            obs[:, I_ANGVEL:I_ANGVEL + 3],      # already scaled by 0.25 in the env
            obs[:, I_GRAV:I_GRAV + 3],
            cmd,
            obs[:, I_QPOS:I_QPOS + 12],
            obs[:, I_QVEL:I_QVEL + 12],         # already scaled by 0.05 in the env
            obs[:, I_ACT:I_ACT + 12],
            obs[:, I_PHASE:I_PHASE + 2],
        ], dim=1)

        h = state_in[:, :H].unsqueeze(0).contiguous()
        c = state_in[:, H:2 * H].unsqueeze(0).contiguous()
        out, (h2, c2) = self.lstm(u.unsqueeze(0), (h, c))
        action = self.actor(out.squeeze(0))
        # Only the first 2H slots are used; the rest of the state vector stays zero.
        state_out = torch.cat([h2.squeeze(0), c2.squeeze(0),
                               torch.zeros_like(state_in[:, 2 * H:])], dim=1)
        return action, state_out


def check(src: torch.jit.ScriptModule, policy: BaselinePolicy, n: int = 64) -> float:
    """Roll both forward on the same random observations; return the max action difference."""
    torch.manual_seed(0)
    state = torch.zeros(1, STATE_DIM)
    worst = 0.0
    with torch.no_grad():
        for _ in range(n):
            obs = torch.randn(1, OBS_DIM) * 0.3
            mine, state = policy(obs, state)
            # Feed the ORIGINAL the same 47-d vector our wrapper built, so the only thing under
            # test is the LSTM+MLP rebuild, not the slicing. Both carry state from step to step:
            # motion.pt in its own buffers, ours in `state`.
            theirs = src(_u_of(policy, obs))
            worst = max(worst, float((mine - theirs).abs().max()))
    return worst


def _u_of(policy: BaselinePolicy, obs: torch.Tensor) -> torch.Tensor:
    """The 47-d motion.pt input our wrapper constructs, recomputed for the equivalence check."""
    sin_yaw, y = obs[:, I_YAW:I_YAW + 1], obs[:, I_Y:I_Y + 1]
    cmd = torch.cat([
        torch.full_like(y, FORWARD_SPEED),
        torch.clamp(-0.5 * y, -0.3, 0.3),
        torch.clamp(-1.5 * sin_yaw - 0.8 * y, -0.6, 0.6),
    ], dim=1) * policy.cmd_scale
    return torch.cat([obs[:, I_ANGVEL:I_ANGVEL + 3], obs[:, I_GRAV:I_GRAV + 3], cmd,
                      obs[:, I_QPOS:I_QPOS + 12], obs[:, I_QVEL:I_QVEL + 12],
                      obs[:, I_ACT:I_ACT + 12], obs[:, I_PHASE:I_PHASE + 2]], dim=1)


def build(urlg: pathlib.Path, out: pathlib.Path) -> None:
    src = torch.jit.load(str(urlg / "deploy/pre_train/g1/motion.pt")).eval()
    policy = BaselinePolicy(src).eval()

    worst = check(src, policy)
    print(f"rebuild vs motion.pt: max |delta action| = {worst:.3e}")
    assert worst < 1e-5, "rebuilt policy does not match motion.pt"

    obs = torch.zeros(1, OBS_DIM)
    state = torch.zeros(1, STATE_DIM)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        policy, (obs, state), str(out),
        input_names=["obs", "state_in"], output_names=["action", "state_out"],
        dynamic_axes={k: {0: "batch"} for k in ("obs", "state_in", "action", "state_out")},
        opset_version=17, dynamo=False)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--urlg", default="unitree_rl_gym")
    ap.add_argument("--out", default="baseline/baseline.onnx")
    a = ap.parse_args()
    build(pathlib.Path(a.urlg), pathlib.Path(a.out))

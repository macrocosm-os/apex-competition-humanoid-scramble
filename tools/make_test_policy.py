"""Emit a small untrained ONNX policy with the competition's exact tensor signature.

Not a baseline — it cannot walk. Its job is to prove the loop is wired correctly: a policy that
conforms to the interface must be driven through real physics and terminate for a real reason,
not for player_error or invalid_action. Release CI gates on that.

It is also the reference for what a conforming graph looks like:

    inputs   obs       float32 [batch, 104]
             state_in  float32 [batch, 256]
    outputs  action    float32 [batch, 12]
             state_out float32 [batch, 256]

    python tools/make_test_policy.py --out /tmp/test_policy.onnx
"""

from __future__ import annotations

import argparse
import pathlib

import torch
from torch import nn

from env.sim import ACT_DIM, OBS_DIM, STATE_DIM

HIDDEN = 64


class TestPolicy(nn.Module):
    """A GRU-ish toy: enough weights to clear screening's min_weight_bytes, and it does use
    `state_in`, so the state round-trip is exercised rather than assumed."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(OBS_DIM, HIDDEN)
        self.mix = nn.Linear(HIDDEN + HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, ACT_DIM)

    def forward(self, obs: torch.Tensor, state_in: torch.Tensor):
        h = torch.tanh(self.enc(obs))
        h = torch.tanh(self.mix(torch.cat([h, state_in[:, :HIDDEN]], dim=1)))
        action = torch.tanh(self.head(h))
        state_out = torch.cat([h, torch.zeros_like(state_in[:, HIDDEN:])], dim=1)
        return action, state_out


def build(out: pathlib.Path, seed: int = 0) -> None:
    torch.manual_seed(seed)
    policy = TestPolicy().eval()
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
    ap.add_argument("--out", default="/tmp/test_policy.onnx")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(pathlib.Path(a.out), a.seed)

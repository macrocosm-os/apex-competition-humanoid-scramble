"""Build a conforming, genuinely-trained walking policy at this competition's 22-DoF interface.

`tools/make_baseline.py` is the 12-DoF/104-obs original and says in its own header that it needs
a full rewrite for the arm interface. This is that rewrite, kept as a separate tool rather than
replacing it: the artifact here is NOT the competition's baseline. `spec.defaults.baseline_raw_score`
stays 0.0 by choice (it is the entry bar, and a bar above the true baseline stalls the round
forever), and nothing about this file changes it.

What it is for: having a policy that actually walks, so the evaluation loop can be exercised over
its real length. An untrained graph (tools/make_test_policy.py) falls within a few hundred control
steps, so it only ever measures the cheap end -- the expensive case is a policy that survives an
instance and spends those steps in contact with boxes it has already disturbed. That is what the
box_scramble load test needs (apex-mvp docs/box-scramble-load-test-plan.md).

Legs come from Unitree's stock G1 walker, `deploy/pre_train/g1/motion.pt` from
unitree_rl_gym (BSD-3) -- a real flat-ground locomotion policy, not trained on this course and
unable to see it. The wrapper does what make_baseline.py's did:

  1. slices the 136-d observation into the 47-d vector motion.pt was trained on, taking the
     LEG half of the 22-value joint blocks (legs are [0:12], see env/sim.py's KP comment),
  2. synthesises the body-frame velocity command it expects, closing a heading-hold loop on the
     room's centreline -- it has no heading feedback of its own,
  3. ignores perception entirely.

The 10 arm outputs are held at the default pose. motion.pt has no arm outputs, and inventing
them would be an untrained guess that makes the robot fall sooner, not later.

    python tools/make_walker22.py --urlg <unitree_rl_gym checkout> --out /tmp/walker22.onnx

Sparse checkout for the weights (~150 KB, no need for the rest of the repo):

    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/unitreerobotics/unitree_rl_gym.git
    cd unitree_rl_gym && git sparse-checkout set deploy/pre_train

Needs torch, which neither image ships -- this runs on a host, like make_test_policy.py.

On artifact SIZE: the load-test plan prefers a ~14 MB graph on the theory that inference cost is
linear in artifact size and 14 MB is the worst case the 15 MB cap admits. Measured on this course
that lever is not worth pulling -- inference is 0.035 ms per control step against 8.05 ms of
physics with ~840 contacts, so padding the graph moves the total by well under a percent. The
0.14 MB graph this writes is the honest worst case for wall-clock purposes.
"""

from __future__ import annotations

import argparse
import pathlib

import torch
from torch import nn

from env.sim import ACT_DIM, N_LEG_DOF, OBS_DIM, STATE_DIM

# Indices into the 136-d observation, in env/sim.py's _obs concatenation order:
#   grav 3 | ang*0.25 3 | lin*2 3 | angles 22 | vel 22 | action 22 | gait 2 | yaw 2 |
#   y,dist,clearance 3 | scan 45 | over 7 | hand 2
# The joint blocks are 22 wide here where upstream parkour's were 12, which is the whole reason
# make_baseline.py's indices do not transfer.
I_GRAV, I_ANGVEL, I_QPOS, I_QVEL, I_ACT, I_PHASE, I_YAW, I_Y = 0, 3, 9, 31, 53, 75, 77, 79

CMD_SCALE = torch.tensor([2.0, 2.0, 0.25])
FORWARD_SPEED = 0.8
H = 64  # motion.pt LSTM width


class Walker22(nn.Module):
    """Stock G1 walker on the legs, default-pose hold on the arms, competition interface."""

    def __init__(self, src: torch.jit.ScriptModule):
        super().__init__()
        self.lstm = nn.LSTM(47, H, batch_first=False)
        self.actor = nn.Sequential(nn.Linear(H, 32), nn.ELU(), nn.Linear(32, N_LEG_DOF))
        sd = dict(src.named_parameters())
        with torch.no_grad():
            for k in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
                getattr(self.lstm, k).copy_(sd[f"memory.{k}"])
            self.actor[0].weight.copy_(sd["actor.0.weight"])
            self.actor[0].bias.copy_(sd["actor.0.bias"])
            self.actor[2].weight.copy_(sd["actor.2.weight"])
            self.actor[2].bias.copy_(sd["actor.2.bias"])
        self.register_buffer("cmd_scale", CMD_SCALE)

    def motion_input(self, obs: torch.Tensor) -> torch.Tensor:
        """The 47-d vector motion.pt was trained on, built from this competition's observation."""
        # sin(yaw) stands in for yaw: the hold loop keeps the error small, where the two agree to
        # well under a degree, and it avoids atan2 in ONNX.
        sin_yaw = obs[:, I_YAW:I_YAW + 1]
        y = obs[:, I_Y:I_Y + 1]
        cmd = torch.cat([
            torch.full_like(y, FORWARD_SPEED),
            torch.clamp(-0.5 * y, -0.3, 0.3),
            torch.clamp(-1.5 * sin_yaw - 0.8 * y, -0.6, 0.6),
        ], dim=1) * self.cmd_scale
        return torch.cat([
            obs[:, I_ANGVEL:I_ANGVEL + 3],          # already scaled by 0.25 in the env
            obs[:, I_GRAV:I_GRAV + 3],
            cmd,
            obs[:, I_QPOS:I_QPOS + N_LEG_DOF],      # leg half of the 22-value block
            obs[:, I_QVEL:I_QVEL + N_LEG_DOF],      # already scaled by 0.05 in the env
            obs[:, I_ACT:I_ACT + N_LEG_DOF],
            obs[:, I_PHASE:I_PHASE + 2],
        ], dim=1)

    def forward(self, obs: torch.Tensor, state_in: torch.Tensor):
        u = self.motion_input(obs)
        h = state_in[:, :H].unsqueeze(0).contiguous()
        c = state_in[:, H:2 * H].unsqueeze(0).contiguous()
        out, (h2, c2) = self.lstm(u.unsqueeze(0), (h, c))
        legs = self.actor(out.squeeze(0))
        arms = torch.zeros_like(legs[:, :ACT_DIM - N_LEG_DOF])   # hold DEFAULT_ANGLES
        action = torch.cat([legs, arms], dim=1)
        # Only the first 2H slots are used; the rest of the state vector stays zero.
        state_out = torch.cat([h2.squeeze(0), c2.squeeze(0),
                               torch.zeros_like(state_in[:, 2 * H:])], dim=1)
        return action, state_out


def check(src: torch.jit.ScriptModule, policy: Walker22, n: int = 64) -> float:
    """Roll both forward on the same observations; return the max LEG action difference.

    The rebuild lifts motion.pt's weights into a plain nn.Module (its TorchScript build hides the
    LSTM state in mutating buffers that neither ONNX exporter will trace), so it is checked rather
    than assumed. Feeds the ORIGINAL the same 47-d vector the wrapper built, so the only thing
    under test is the LSTM+MLP rebuild, not the slicing."""
    torch.manual_seed(0)
    state = torch.zeros(1, STATE_DIM)
    worst = 0.0
    with torch.no_grad():
        for _ in range(n):
            obs = torch.randn(1, OBS_DIM) * 0.3
            mine, state = policy(obs, state)
            theirs = src(policy.motion_input(obs))
            worst = max(worst, float((mine[:, :N_LEG_DOF] - theirs).abs().max()))
    return worst


def build(urlg: pathlib.Path, out: pathlib.Path) -> None:
    src = torch.jit.load(str(urlg / "deploy/pre_train/g1/motion.pt")).eval()
    policy = Walker22(src).eval()

    worst = check(src, policy)
    print(f"legs vs motion.pt: max |delta action| = {worst:.3e}")
    assert worst < 1e-5, "rebuilt legs do not match motion.pt"

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
    ap.add_argument("--urlg", default="unitree_rl_gym", help="unitree_rl_gym checkout")
    ap.add_argument("--out", default="/tmp/walker22.onnx")
    a = ap.parse_args()
    build(pathlib.Path(a.urlg), pathlib.Path(a.out))

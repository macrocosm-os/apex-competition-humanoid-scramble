"""Measure ONNX inference latency vs model size, single-threaded, at the competition's signature.

This is the evidence behind `submission.max_size_mb` (docs/design.md). The cap is only defensible
if a model at the limit still fits the referee's time budget on the hardware the platform actually
runs, so this is meant to be run in CI on a real worker-class CPU, not just on a dev machine.

    python tools/latency_probe.py
    python tools/latency_probe.py --instances 24 --max-steps 4000 --physics-s 109
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from env.sim import ACT_DIM, OBS_DIM, STATE_DIM

ARCHS = [(64, 32), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
REFEREE_TIMEOUT_S = 900


def _export(widths: tuple[int, ...], path: str) -> int:
    layers: list[nn.Module] = []
    d = OBS_DIM + STATE_DIM
    for w in widths:
        layers += [nn.Linear(d, w), nn.ELU()]
        d = w

    class P(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(d, ACT_DIM)
            self.st = nn.Linear(d, STATE_DIM)

        def forward(self, obs, state_in):
            h = self.body(torch.cat([obs, state_in], 1))
            return self.head(h), torch.tanh(self.st(h))

    p = P().eval()
    torch.onnx.export(
        p, (torch.zeros(1, OBS_DIM), torch.zeros(1, STATE_DIM)), path,
        input_names=["obs", "state_in"], output_names=["action", "state_out"],
        dynamic_axes={k: {0: "batch"} for k in ("obs", "state_in", "action", "state_out")},
        opset_version=17, dynamo=False)
    return sum(q.numel() for q in p.parameters())


def probe(instances: int, max_steps: int, physics_s: float, cap_mb: float) -> None:
    calls = instances * max_steps
    budget_ms = (REFEREE_TIMEOUT_S - physics_s) / calls * 1000
    print(f"{calls:,} inferences per full-survival run; "
          f"{REFEREE_TIMEOUT_S - physics_s:.0f} s of slack -> {budget_ms:.2f} ms/inference\n")
    print(f"{'arch':>22} {'params':>12} {'MB':>7} {'ms/step':>9} {'full run':>10}  verdict")

    worst_ok_mb = 0.0
    for widths in ARCHS:
        path = tempfile.mktemp(suffix=".onnx")
        try:
            n = _export(widths, path)
            mb = os.path.getsize(path) / 1048576
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = opts.inter_op_num_threads = 1
            s = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
            names = [i.name for i in s.get_inputs()]
            obs = np.zeros((1, OBS_DIM), np.float32)
            st = np.zeros((1, STATE_DIM), np.float32)
            for _ in range(20):
                s.run(None, {names[0]: obs, names[1]: st})
            t = time.perf_counter()
            for _ in range(300):
                s.run(None, {names[0]: obs, names[1]: st})
            ms = (time.perf_counter() - t) / 300 * 1000
            ok = ms < budget_ms
            if ok:
                worst_ok_mb = max(worst_ok_mb, mb)
            print(f"{str(widths):>22} {n:>12,} {mb:>7.2f} {ms:>9.3f} "
                  f"{ms * calls / 1000:>9.0f}s  {'ok' if ok else 'TOO SLOW'}")
        finally:
            if os.path.exists(path):
                os.remove(path)

    print(f"\nlargest architecture inside the budget on this CPU: {worst_ok_mb:.0f} MB")
    print(f"spec cap: {cap_mb:.0f} MB", end="  ")
    if worst_ok_mb >= cap_mb:
        print(f"-> the cap is NOT compute-bound (headroom {worst_ok_mb / cap_mb:.1f}x)")
    else:
        print("-> WARNING: the cap admits models this CPU cannot run in budget")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=24)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--physics-s", type=float, default=109.0,
                    help="measured physics + HTTP cost of a full-survival run")
    ap.add_argument("--cap-mb", type=float, default=25.0)
    a = ap.parse_args()
    probe(a.instances, a.max_steps, a.physics_s, a.cap_mb)

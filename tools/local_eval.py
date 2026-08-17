"""Score an ONNX policy against the course, in-process (no player/referee containers).

Same env, same gates, same scoring function the referee uses, so the numbers move together —
but this skips HTTP, so it is for calibration and variance measurement, not for producing the
figure that goes in spec.yaml. That one has to be measured inside the referee image.

    python tools/local_eval.py baseline/baseline.onnx -n 20
    python tools/local_eval.py baseline/baseline.onnx -n 20 --seed 7 --json out.json
    python tools/local_eval.py baseline/baseline.onnx -n 20 --seed 7 --record runs/base

Conditions are drawn from --seed, so raw_score moves between seeds. Sweep it to measure
round-to-round variance.

`--record DIR` writes the same per-instance history files the referee writes to /data/history/ —
one `instance_NN.json` each, replayable with tools/replay.py (format in env/history.py). It costs
an array copy per control step and changes nothing about the scoring, so a recorded suite scores
identically to an unrecorded one.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import numpy as np
import onnxruntime as ort

from env import ParkourSim, instance_score, instance_spec
from env.history import DEFAULT_STRIDE, InstanceRecorder, write_instance
from env.sim import FRAME_SKIP, OBS_DIM, PHYS_DT, STATE_DIM


def rollout(session, sim: ParkourSim, max_steps: int, rec: InstanceRecorder | None = None):
    obs = sim.reset()
    state = np.zeros((1, STATE_DIM), np.float32)
    names = [i.name for i in session.get_inputs()]
    reason = None
    while reason is None:
        action, state = session.run(None, {names[0]: obs.reshape(1, OBS_DIM), names[1]: state})
        action = np.asarray(action).ravel()
        result = sim.step(action, max_steps=max_steps)
        obs, reason = result.obs, result.terminal_reason
        if rec is not None:
            rec.capture(sim, action)
    return reason


def evaluate(path: str, n: int, max_steps: int, seed: int = 1, verbose: bool = True,
             record: str | None = None, stride: int = DEFAULT_STRIDE):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])

    rows, written, t0 = [], [], time.monotonic()
    for i in range(n):
        params = instance_spec(i, n, seed)
        sim = ParkourSim(params)
        rec = InstanceRecorder(i, sim, stride) if record else None
        reason = rollout(session, sim, max_steps, rec)
        score = instance_score(reason, sim.progress, sim.steps, max_steps)
        rows.append({"instance": i, "friction_level": round(params.friction_level, 4),
                     "wind_speed_ms": round(params.wind_speed, 2), "terminal_reason": reason,
                     "progress": round(sim.progress, 4), "steps": sim.steps,
                     "score": round(score, 4), "max_x": round(sim.max_x, 2),
                     # The referee's key for the same quantity; history files carry both names.
                     "distance_m": round(sim.max_x, 2),
                     "sim_time_s": round(sim.steps * PHYS_DT * FRAME_SKIP, 2)})
        if rec is not None:
            written.append(write_instance(record, rec.record(
                sim, rows[-1], match_id=f"local:{pathlib.Path(path).stem}", num_instances=n)))
        if verbose:
            print(f"  [{i + 1:3d}/{n}] mu {params.friction_level:.3f} wind "
                  f"{params.wind_speed:4.1f} m/s  {reason:14s} {sim.max_x:6.2f} m  "
                  f"progress {sim.progress:.3f}  score {score:.3f}")

    scores = [r["score"] for r in rows]
    summary = {
        "artifact": path, "n": n, "max_steps": max_steps, "seed": seed,
        "raw_score": round(statistics.fmean(scores), 4),
        "stdev": round(statistics.stdev(scores), 4) if n > 1 else 0.0,
        "sem": round(statistics.stdev(scores) / n ** 0.5, 4) if n > 1 else 0.0,
        "num_completed": sum(r["terminal_reason"] == "completed" for r in rows),
        "mean_max_x": round(statistics.fmean(r["max_x"] for r in rows), 2),
        "reasons": {r: sum(x["terminal_reason"] == r for x in rows)
                    for r in sorted({x["terminal_reason"] for x in rows})},
        "wall_time_s": round(time.monotonic() - t0, 1),
        "instances": rows,
    }
    if written:
        summary["history_dir"] = str(record)
        summary["history_files"] = len(written)
        if verbose:
            total = sum(p.stat().st_size for p in written)
            print(f"wrote {len(written)} history files to {record}/ ({total / 1e6:.1f} MB)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1, help="round seed: sets friction and wind")
    ap.add_argument("--json")
    ap.add_argument("--record", metavar="DIR",
                    help="write per-instance history files here — see tools/replay.py")
    ap.add_argument("--record-stride", type=int, default=DEFAULT_STRIDE, metavar="N",
                    help=f"keep every Nth control step (default {DEFAULT_STRIDE} = 25 Hz); "
                         "1 records all 50 Hz")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    s = evaluate(a.artifact, a.n, a.max_steps, seed=a.seed, verbose=not a.quiet,
                 record=a.record, stride=a.record_stride)
    print(json.dumps({k: v for k, v in s.items() if k != "instances"}, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(s, f, indent=2)
        print("wrote", a.json)

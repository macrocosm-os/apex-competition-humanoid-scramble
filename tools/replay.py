"""Play back a recorded evaluation. MuJoCo only — no policy, no onnxruntime, no physics.

    PYTHONPATH=. python tools/replay.py runs/base              # list what is in the history
    PYTHONPATH=. python tools/replay.py runs/base -i 7         # film instance 7 to mp4
    PYTHONPATH=. python tools/replay.py runs/base --worst      # film the lowest-scoring one
    PYTHONPATH=. python tools/replay.py runs/base --all        # film every instance
    PYTHONPATH=. mjpython tools/replay.py runs/base -i 7 --live   # interactive viewer

Takes a directory of `instance_NN.json` files or a single one, so it reads a local recording
(`local_eval.py --record DIR`) and a history artifact downloaded from a scored round identically —
same format either way (env/history.py).

Replay sets `data.qpos` from the file and calls `mj_forward`, which recomputes every derived
quantity a renderer needs. Nothing is integrated, so a replay cannot drift from the scored run the
way re-simulating from the action log could — and it needs neither the submission nor the round
seed, so a run stays viewable after both are gone.

The scene is rebuilt from the frictions stored per instance, through the same `_lit_model` the
preview uses, so it is the scored geometry with lights added and nothing else changed.

`--live` needs `mjpython` on macOS rather than `python`: the passive viewer has to own the main
thread there. Filming to mp4 is fully headless and has no such constraint. Needs ffmpeg.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import mujoco

from env.course import ROOM_LENGTH
from env.history import read_all, unpack
from tools.preview import OUT, _camera, _lit_model, frames_dir, mp4, png

TARGET_FPS = 30.0


class Run:
    """One instance's history, with the arrays unpacked."""

    def __init__(self, record: dict):
        self.record = record
        self.index = int(record["instance"])
        self.outcome = record.get("outcome", {})
        frames = record["frames"]
        self.qpos = unpack(frames["qpos"])
        self.ticks = unpack(frames["ticks"])
        self.action = unpack(frames["action"])
        self.round_seed = record["conditions"]["round_seed"]
        timing = record.get("timing", {})
        self.frame_dt = float(timing.get("control_dt", 0.02)) * int(timing.get("stride", 1))

    @property
    def frames(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def score(self) -> float:
        return float(self.outcome.get("score") or 0.0)

    def describe(self) -> str:
        o, c = self.outcome, self.record["conditions"]
        return (f"instance {self.index:3d}  "
                f"wind {float(c.get('wind_speed_ms') or 0):4.1f} m/s @ "
                f"{float(c.get('wind_dir_deg') or 0):5.1f}deg  "
                f"{str(o.get('terminal_reason')):14s} {float(o.get('distance_m') or 0):6.2f} m "
                f"of {ROOM_LENGTH:.1f} m  score {self.score:.4f}  {self.frames} frames "
                f"({max(self.frames - 1, 0) * self.frame_dt:.1f} s)")


def _chase(cam, qpos) -> None:
    """The same over-the-shoulder framing `tools/preview.py --run` uses, for comparability."""
    cam.lookat[:] = [float(qpos[0]) + 1.2, 0, float(qpos[2])]
    cam.distance, cam.azimuth, cam.elevation = 4.6, 128, -12


def _stride_for(frame_dt: float, target_fps: float) -> tuple[int, float]:
    """Frames to skip so playback runs at real speed near `target_fps`, and the fps that gives."""
    step = max(1, int(round((1.0 / target_fps) / frame_dt)))
    return step, 1.0 / (frame_dt * step)


def _scene(run: Run) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Rebuild the instance's scene from its recorded round seed, refusing a mismatched history."""
    model = _lit_model(run.round_seed)
    if run.qpos.shape[1] != model.nq:
        raise ValueError(f"history has nq={run.qpos.shape[1]}, this model has nq={model.nq}; "
                         "the file predates a model change and cannot be replayed against it")
    return model, mujoco.MjData(model)


def film(run: Run, out: pathlib.Path, target_fps: float = TARGET_FPS,
         width: int = 1280, height: int = 720) -> None:
    model, data = _scene(run)
    cam, opt = _camera()
    renderer = mujoco.Renderer(model, height=height, width=width)
    step, fps = _stride_for(run.frame_dt, target_fps)

    fd = frames_dir(f"_replay_{run.index:03d}")
    keep = list(range(0, run.frames, step))
    if not keep:
        raise ValueError(f"instance {run.index} has no recorded frames")
    if keep[-1] != run.frames - 1:
        keep.append(run.frames - 1)   # always end on the terminal pose
    for fi, src in enumerate(keep):
        data.qpos[:] = run.qpos[src]
        mujoco.mj_forward(model, data)
        _chase(cam, data.qpos)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        png(renderer.render(), fd / f"f{fi:05d}.png")
    mp4(fd, out, fps=max(1, round(fps)))   # a heavily strided history can round below 1 fps


def live(run: Run, speed: float = 1.0, loop: bool = False) -> None:
    import mujoco.viewer

    model, data = _scene(run)
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        raise SystemExit(f"{e}\n\nOn macOS the passive viewer must own the main thread — run this "
                         f"with `mjpython tools/replay.py ...` instead of `python`.") from e
    dt = run.frame_dt / max(speed, 1e-6)
    with viewer as v:
        while True:
            # Pace against a wall-clock deadline rather than sleeping a flat dt, so the time spent
            # in mj_forward and sync does not quietly stretch playback past real time.
            deadline = time.monotonic()
            for frame in run.qpos:
                if not v.is_running():
                    return
                data.qpos[:] = frame
                mujoco.mj_forward(model, data)
                v.sync()
                deadline += dt
                time.sleep(max(0.0, deadline - time.monotonic()))
            if not loop:
                return


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay a recorded evaluation with MuJoCo alone.")
    ap.add_argument("history", help="directory of instance_NN.json files, or one such file")
    ap.add_argument("-i", "--instance", type=int, help="which instance to play back")
    ap.add_argument("--worst", action="store_true", help="pick the lowest-scoring instance")
    ap.add_argument("--all", action="store_true", help="film every instance in the history")
    ap.add_argument("--live", action="store_true",
                    help="interactive viewer instead of mp4 (needs mjpython on macOS)")
    ap.add_argument("--speed", type=float, default=1.0, help="--live playback rate; 1.0 is real time")
    ap.add_argument("--loop", action="store_true", help="--live: repeat until the window is closed")
    ap.add_argument("--fps", type=float, default=TARGET_FPS, help="mp4 frame rate to aim for")
    ap.add_argument("--out", help="mp4 path (single instance only; default renders/replay_NNN.mp4)")
    a = ap.parse_args()

    runs = [Run(r) for r in read_all(a.history)]
    first = runs[0].record
    print(f"{a.history}: {len(runs)} instances of {first.get('num_instances')}, "
          f"match {first.get('match_id')}, mujoco {first.get('mujoco_version')}, "
          f"{1 / runs[0].frame_dt:.0f} Hz (stride {first.get('timing', {}).get('stride')})")

    if a.all:
        OUT.mkdir(exist_ok=True)
        for run in runs:
            print(" ", run.describe())
            film(run, OUT / f"replay_{run.index:03d}.mp4", target_fps=a.fps)
        raise SystemExit(0)

    if a.instance is None and not a.worst:
        for run in runs:
            print(" ", run.describe())
        print("\nPass -i N to film one, --worst for the lowest-scoring, or --all for every one.")
        raise SystemExit(0)

    if a.worst:
        chosen = min(runs, key=lambda r: r.score)
    else:
        matches = [r for r in runs if r.index == a.instance]
        if not matches:
            have = ", ".join(str(r.index) for r in runs)
            raise SystemExit(f"instance {a.instance} is not in this history; it has [{have}]")
        chosen = matches[0]
    print(" ", chosen.describe())

    if a.live:
        live(chosen, speed=a.speed, loop=a.loop)
    else:
        OUT.mkdir(exist_ok=True)
        film(chosen, pathlib.Path(a.out) if a.out
             else OUT / f"replay_{chosen.index:03d}.mp4", target_fps=a.fps)

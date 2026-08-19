"""Render the course: stills, a flythrough, and a run by an actual policy.

    python tools/preview.py                              # stills + flythrough
    python tools/preview.py --run baseline/baseline.onnx  # + a scored run, filmed

Everything renders against the same vendored G1 and the same `ParkourSim` the referee uses, so
what you see is what gets scored — there is no separate preview model to drift out of sync, and
no external checkout to clone. Needs mujoco + ffmpeg (and onnxruntime for --run).
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import subprocess

import mujoco
import numpy as np

from env import ParkourSim, instance_score, instance_spec
from env.course import PLINTH_TOP, ROOM_LENGTH, TRACK_HALF_W
from env.sim import OBS_DIM, STATE_DIM, START_X, _mesh_assets

OUT = pathlib.Path("renders")


def _lit_model(round_seed: int) -> mujoco.MjModel:
    """The scored scene for a given round seed, re-compiled with lighting and a framebuffer big
    enough to render. For a recorded run use `_lit_model_for_boxes` instead -- a replay works from
    the box field the history stored, not from a seed.
    """
    from env.course import build_course
    rng = np.random.default_rng([round_seed, 0xB0B])
    _, boxes, _ = build_course(rng)
    return _lit_model_for_boxes(boxes)


def _lit_model_for_boxes(boxes: list) -> mujoco.MjModel:
    """The scored scene for a given box field, re-compiled with lighting and a framebuffer big
    enough to render.

    The robot XML is authored for headless physics: no lights worth the name, a black backdrop,
    and a 640x480 offscreen buffer. None of that changes the dynamics, so the preview compiles a
    cosmetically patched copy rather than polluting the model the referee runs.

    Takes the box field itself (not a `ParkourSim`, not a seed) so `tools/replay.py` can rebuild
    the scored scene from a recording's stored conditions, with no sim and no policy in reach.
    """
    # _round_scene (env/sim.py) is the scored path but returns a compiled MjModel with no MJCF
    # string to patch lighting into; re-emit the identical XML here from the same box field
    # (same boxes -> same scene, bit for bit) rather than re-serialising.
    from env.course import boxes_xml_fragment, build_floor, floor_xml_fragment
    from env.sim import _scene_xml
    xml = _scene_xml(floor_xml_fragment(build_floor()), boxes_xml_fragment(boxes))
    # The schema permits exactly one <global>, and the robot already has one, so extend it.
    xml = re.sub(r"<global\b", '<global offwidth="1600" offheight="900"', xml, count=1)
    if "<visual>" not in xml:
        xml = xml.replace("<worldbody>", "<visual><global offwidth='1600' offheight='900'/>"
                          "</visual>\n  <worldbody>", 1)
    # The referee's own floor (env/sim._scene_xml) is an 80 m plane centred at x=30 -- it needs
    # to extend well past ROOM_LENGTH so a robot that overshoots after "completed" doesn't ray-
    # miss the floor, but that same slab makes the PREVIEW look like the course keeps going
    # forever past the finish line (Crux, 2026-08-18: "why is the yellow finish line not at the
    # end of the course" -- it IS at x=ROOM_LENGTH, the visible floor just doesn't stop there).
    # Cosmetic fix only: cap the floor a couple metres past the finish line for RENDERING, and
    # cap it with an actual visible end-cap wall so the course reads as finite on camera. Does
    # not touch scoring (referee.py / sim.py's floor plane and termination gate are untouched).
    floor_end = ROOM_LENGTH + 0.4   # small ray-safety margin only -- tight enough that the
                                     # end-cap wall reads as right behind the finish line, not a
                                     # separate wall further down the platform (Crux, 2026-08-18:
                                     # the first cap attempt left too much floor between the
                                     # yellow line and the wall)
    xml = re.sub(
        r'<geom name="floor" type="plane" size="80 20 0.1" pos="30 0 0"([^/]*)/>',
        f'<geom name="floor" type="plane" size="{floor_end / 2:.2f} {TRACK_HALF_W + 1:.2f} 0.1" '
        f'pos="{floor_end / 2:.2f} 0 0"\\1/>',
        xml, count=1,
    )
    xml = xml.replace("<worldbody>",
        '<worldbody>\n'
        '    <light pos="8 -6 9" dir="-0.3 0.4 -1" directional="true" diffuse="0.6 0.6 0.57"/>\n'
        '    <light pos="34 6 9" dir="0.2 -0.4 -1" directional="true" diffuse="0.35 0.36 0.40"/>\n'
        # End-cap wall just past the finish line so the platform visibly terminates on camera,
        # instead of fading into open black background -- makes "this is the end" legible from
        # any camera angle, not just ones that happen to frame the yellow line clearly.
        f'    <geom type="box" pos="{floor_end:.2f} 0 {PLINTH_TOP:.2f}" '
        f'size="0.05 {TRACK_HALF_W + 1:.2f} {PLINTH_TOP:.2f}" group="2" rgba=".30 .32 .35 1"/>\n'
        f'    <geom type="box" pos="{ROOM_LENGTH:.2f} 0 {PLINTH_TOP + 0.9:.2f}" '
        f'size="0.04 {TRACK_HALF_W:.2f} 0.02" group="2" rgba="1 .95 .2 1"/>', 1)
    return mujoco.MjModel.from_xml_string(xml, _mesh_assets())


def png(px, path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{px.shape[1]}x{px.shape[0]}", "-i", "-", str(path)],
                   input=px.tobytes(), check=True)


def mp4(frame_dir, out, fps=30):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(frame_dir / "f%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True)
    print("wrote", out)


def frames_dir(name):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.png"):
        f.unlink()
    return d


def _camera():
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)
    return cam, opt


def render_course(sim: ParkourSim):
    m = _lit_model(sim.params.round_seed)
    d = mujoco.MjData(m)
    d.qpos[0], d.qpos[2] = START_X, PLINTH_TOP + 0.793
    mujoco.mj_forward(m, d)
    cam, opt = _camera()

    r = mujoco.Renderer(m, height=900, width=1600)
    for name, look, dist, elev in [("01_overview_iso", ROOM_LENGTH / 2, 44, -22),
                                   ("02_side_elevation", ROOM_LENGTH / 2, 40, -6),
                                   ("03_scramble_field", 3.5, 14, -18),
                                   ("04_push_corridor", 6.5, 13, -18),
                                   ("05_climb_stack", 9.5, 13, -16),
                                   ("06_dash_finish", 11.5, 15, -14)]:
        cam.lookat[:] = [look, 0, 1.2]
        cam.distance, cam.azimuth, cam.elevation = dist, (90 if "side" in name else 132), elev
        r.update_scene(d, camera=cam, scene_option=opt)
        png(r.render(), OUT / f"{name}.png")
    print(f"wrote 6 stills to {OUT}/")

    fd = frames_dir("_fly")
    r = mujoco.Renderer(m, height=720, width=1280)
    n = 300
    for i in range(n):
        t = i / (n - 1)
        cam.lookat[:] = [-2.0 + t * (ROOM_LENGTH + 4.0), 0, 1.0]
        cam.distance, cam.elevation = 9.0, -13 - 6 * math.sin(math.pi * t)
        cam.azimuth = 118 + 24 * math.sin(2 * math.pi * t)
        r.update_scene(d, camera=cam, scene_option=opt)
        png(r.render(), fd / f"f{i:05d}.png")
    mp4(fd, OUT / "course_flythrough.mp4")


def film_run(sim: ParkourSim, artifact: str, max_steps: int):
    """Drive the real sim with a real ONNX policy and film it. Scores exactly as the referee does."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(artifact, sess_options=opts, providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]

    # Physics runs on the sim's own model; the camera renders a cosmetically lit twin that shares
    # its geometry, so qpos transfers across directly.
    view = _lit_model(sim.params.round_seed)
    vdata = mujoco.MjData(view)
    r = mujoco.Renderer(view, height=720, width=1280)
    cam, opt = _camera()
    fd = frames_dir("_run")

    obs = sim.reset()
    state = np.zeros((1, STATE_DIM), np.float32)
    every = max(1, int(round((1 / 30) / (sim.model.opt.timestep * 10))))
    fi, reason = 0, None
    while reason is None:
        action, state = session.run(None, {names[0]: obs.reshape(1, OBS_DIM), names[1]: state})
        result = sim.step(np.asarray(action).ravel(), max_steps=max_steps)
        obs, reason = result.obs, result.terminal_reason
        if sim.steps % every == 0 or reason is not None:
            vdata.qpos[:] = sim.data.qpos
            mujoco.mj_forward(view, vdata)
            x, z = float(sim.data.qpos[0]), float(sim.data.qpos[2])
            cam.lookat[:] = [x + 1.2, 0, z]
            cam.distance, cam.azimuth, cam.elevation = 4.6, 128, -12
            r.update_scene(vdata, camera=cam, scene_option=opt)
            png(r.render(), fd / f"f{fi:05d}.png"); fi += 1

    score = instance_score(reason, sim.progress, sim.steps, max_steps)
    print(f"{artifact}: {reason} at {sim.max_x:.2f} m of {ROOM_LENGTH:.1f} m "
          f"({100 * sim.progress:.0f}%), score {score:.4f}")
    mp4(fd, OUT / "baseline_run.mp4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="ONNX policy to film driving the course")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--of", type=int, default=24, help="suite size the instance is drawn from")
    ap.add_argument("--seed", type=int, default=1, help="round seed: sets friction and wind")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--no-stills", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    params = instance_spec(a.instance, a.of, a.seed)
    print(f"instance {a.instance} of {a.of} at seed {a.seed}: friction level "
          f"{params.friction_level:.3f}, wind {params.wind_speed:.1f} m/s")
    sim = ParkourSim(params)
    if not a.no_stills:
        render_course(sim)
    if a.run:
        film_run(sim, a.run, a.max_steps)

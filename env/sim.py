"""MuJoCo simulation of one Box Scramble episode.

Fork of Humanoid Parkour's sim.py. Robot, action/PD-loop, wind model, and most gates are
unchanged from upstream (see docs/design.md for the full audit trail on those). What's forked is
the scene itself: instead of a fixed sequence of named maneuvers, the robot crosses a 12 m x 6 m
room scattered with a per-round-sampled field of loose boxes (env/course.py) that are free bodies,
not welded geoms — pushing and climbing are genuine contact-solver outcomes.

Robot: Unitree G1, 12 actuated leg DoF (`env/assets/g1_12dof.xml`, vendored from unitree_rl_gym,
BSD-3). The upper body is real mass and real collision geometry welded to the pelvis — the arms
are there and they hit things, they just are not actuated (see docs/design.md, "Robot: G1 12-DoF,
and why not 29" for why this is a one-way door). Torque actuators driven by a PD loop in `step`,
so the ACTION IS A JOINT POSITION TARGET, not a torque.

Deterministic by construction: same (round seed, instance seed, action sequence) -> same
trajectory. The referee owns the physics; the player only ever sees the observation vector.

Observation (float32, OBS_DIM = 104), all in the robot's yaw frame unless noted:
    [0:3]     projected gravity                        (which way is down, from the body)
    [3:6]     base angular velocity      * 0.25
    [6:9]     base linear velocity       * 2.0
    [9:21]    joint angles - default pose
    [21:33]   joint velocities           * 0.05
    [33:45]   previous action
    [45:47]   gait clock, sin/cos of a 0.8 s cycle
    [47:49]   heading error, sin/cos of yaw (room runs along +x)
    [49]      lateral offset y
    [50]      distance to the finish line / 10
    [51]      pelvis height above the surface directly below
    [52:97]   height scan, 9 x 5 grid, surface height relative to the pelvis
    [97:104]  overhead clearance, 7 samples ahead

The height scan hits WHATEVER is directly below each ray -- floor or a box top, whichever is
higher, exactly like a real height-map sensor. There is no separate "box" channel: a stack reads
as a tall floor. This is the fork's one perceptual difference worth stating plainly, because it
is also the one deliberate simplification -- see docs/design.md, "What the policy can and cannot
see" for why an unlabelled height field is enough to express dash/scramble/push/climb and does
not leak which zone is which.

Action (float32, ACT_DIM = 12): joint position targets as offsets from the default pose,
scaled by ACTION_SCALE and clipped to the joint limits.

The box field (sizes, densities, positions) is drawn once per ROUND from the round seed and is
IDENTICAL for every instance in that round; the box field is also NOT itself observable as
metadata (only sensed through the height scan and physical contact) so a policy has to perceive
and push/climb rather than read a manifest. Per-instance wind still varies, same as upstream. See
env/course.sample_boxes and `RoundScene` below.

Termination gates (each maps to a terminal_reason the miner sees post-round):
    completed       pelvis past the far wall (x >= ROOM_LENGTH)
    fell            pelvis under FALL_CLEARANCE above the surface below it, or torso past ~66 deg
    out_of_bounds   |y| > TRACK_HALF_W (no walking around the room through a wall)
    physics_glitch  NaN/Inf state or |qvel| > 100 (glitch-surfing scores 0)
    timeout         max_steps control steps elapsed
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import mujoco
import numpy as np

from .course import (BOX_PREFIX, N_BOXES, PLINTH_TOP, ROOM_LENGTH, TRACK_HALF_W, WORLD_GROUP,
                     boxes_xml_fragment, build_course, floor_xml_fragment)

ASSETS = pathlib.Path(__file__).parent / "assets"

ACT_DIM = 12

# Opaque per-episode policy memory, threaded by the player between /act calls and zeroed on
# /reset. Recurrence matters here for the same reason it did upstream (wind is unobservable), and
# additionally because box CONTACT STATE (has this crate already been shoved out of the way; is
# the robot mid-climb on a stack) has no dedicated observation channel either -- a policy has to
# remember what it just did to a box, not just what the box currently measures as.
STATE_DIM = 256

SCAN_NX, SCAN_NY = 9, 5          # height-scan grid, 45 rays
OVERHEAD_N = 7
OBS_DIM = 52 + SCAN_NX * SCAN_NY + OVERHEAD_N   # = 104

# Where the scan looks, in metres in the robot's yaw frame. Backwards a little so the policy can
# see the edge it is standing on, forwards far enough to plan a mount onto a box or stack.
SCAN_X = np.linspace(-0.4, 1.6, SCAN_NX)
SCAN_Y = np.linspace(-0.5, 0.5, SCAN_NY)
OVERHEAD_X = np.linspace(0.0, 1.8, OVERHEAD_N)
SCAN_CLIP = 1.5                  # raised from parkour's 1.0 m: climb-zone stacks run up to
                                  # ~1.3 m tall, so the scan needs headroom to report a stack top
                                  # relative to a pelvis that is still standing on the floor below.

PHYS_DT = 0.002
FRAME_SKIP = 10                  # 10 x 0.002 s = 20 ms per control step (50 Hz)
DEFAULT_MAX_STEPS = 4000         # 80 s of sim time
ACTION_SCALE = 0.25
QVEL_GLITCH_LIMIT = 100.0
RESET_NOISE = 0.01
FALL_CLEARANCE = 0.45            # pelvis this far above the surface below, or it has fallen
UPRIGHT_MIN = 0.40               # projected-gravity z; ~66 deg of tilt
RAY_FROM_ABOVE = 4.0             # raised from parkour's 3.0 m: must clear the tallest possible
                                  # climb stack (~1.3 m) plus the pelvis's own standing height

# Unitree's PD gains and home pose for this robot (deploy/deploy_mujoco/configs/g1.yaml).
KP = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], np.float64)
KD = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], np.float64)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], np.float64)

START_X = -0.8
GAIT_PERIOD = 0.8

# Wind, via MuJoCo's inertia-box fluid model: opt.wind is subtracted from each body's linear
# velocity and quadratic drag follows, so it only bites with opt.density > 0. Air at 20 C.
# Unchanged from upstream parkour -- see docs/design.md "Wind" for the full derivation.
AIR_DENSITY = 1.204
WIND_MAX_MS = 8.0


class InvalidAction(ValueError):
    """The action was not a finite ACT_DIM vector."""


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray
    terminal_reason: str | None  # None while the episode is still running


@dataclass(frozen=True)
class InstanceParams:
    """The randomised conditions of one evaluation instance."""

    seed: int                # per-instance episode seed: reset noise
    round_seed: int          # the ROUND seed: the box field is a function of this alone
    wind_speed: float        # m/s
    wind_dir: float          # radians; the direction the wind blows FROM, about +x

    @property
    def wind(self) -> tuple[float, float, float]:
        """World-frame air velocity for opt.wind. Horizontal only, as ground-level wind is.

        dir 0 is a headwind for a robot travelling +x (air moves -x); dir pi/2 blows from +y and
        pushes it toward -y.
        """
        return (-self.wind_speed * float(np.cos(self.wind_dir)),
                -self.wind_speed * float(np.sin(self.wind_dir)), 0.0)


def instance_spec(i: int, n: int, seed: int,
                  wind_max: float = WIND_MAX_MS) -> InstanceParams:
    """The conditions of evaluation instance `i` of `n`, drawn from the round `seed`.

    The box field itself is drawn once from `seed` alone (see `_round_scene`) and is identical
    for every instance in the round -- only wind (and reset noise) vary per instance, exactly
    mirroring upstream parkour's split between "what's fixed all round" (there: geometry: here:
    geometry) and "what's drawn per instance" (there: friction+wind; here: wind only, because
    this course's friction comes from each box's sampled density instead of a course-wide level).

    `seed` has no default on purpose: forgetting it would silently freeze the box field, the
    exact failure upstream's docstring warns about. `n` is deliberately NOT an input to the draw.
    """
    rng = np.random.default_rng([seed, i, 0x5EED])
    return InstanceParams(
        seed=int(rng.integers(1 << 31)),
        round_seed=int(seed),
        wind_speed=float(rng.uniform(0.0, wind_max)),
        wind_dir=float(rng.uniform(0.0, 2.0 * np.pi)),
    )


def _scene_xml(floor_frag: str, boxes_frag: str) -> str:
    """The robot model with the room's floor and box field spliced into its worldbody."""
    robot = (ASSETS / "g1_12dof.xml").read_text()
    wall = (f'    <geom name="floor" type="plane" size="80 20 0.1" pos="30 0 0" '
            f'condim="3" group="{WORLD_GROUP}" rgba=".18 .19 .22 1"/>\n')
    start = (f'    <geom type="box" pos="{START_X - 0.7:.3f} 0 {PLINTH_TOP - 0.2:.3f}" '
             f'size="1.5 {TRACK_HALF_W} 0.2" condim="3" group="{WORLD_GROUP}" '
             f'friction="1 .1 .1" rgba=".45 .47 .5 1"/>\n')
    static_body = wall + start + floor_frag
    xml = robot.replace("</worldbody>", static_body + "\n" + boxes_frag + "\n  </worldbody>")
    return xml


# The scene must be recompiled whenever the box field changes, unlike upstream parkour where
# only friction (a runtime field on a fixed-geometry model) varied between rounds. Box SIZE and
# PLACEMENT are compile-time (MJCF geom size/pos), so there is one compiled MjModel per ROUND
# SEED, cached and reused across every instance within that round -- the round-level analogue of
# parkour's "compile once, reuse across instances" optimisation (docs/design.md, "The scene is
# compiled once, not per instance"). A single referee process only ever evaluates one round, so
# the cache holds exactly one entry in practice; keeping it a dict (not a single slot) just means
# a local tool that walks multiple seeds in one process doesn't recompile needlessly either.
_MODEL_CACHE: dict[int, tuple[mujoco.MjModel, list[int]]] = {}


def _round_scene(round_seed: int) -> tuple[mujoco.MjModel, list[int]]:
    """Compile (or fetch) the model for this round's box field. Returns (model, box_body_ids)."""
    cached = _MODEL_CACHE.get(round_seed)
    if cached is not None:
        return cached
    rng = np.random.default_rng([round_seed, 0xB0B])
    floor_segs, boxes, _ = build_course(rng)
    xml = _scene_xml(floor_xml_fragment(floor_segs), boxes_xml_fragment(boxes))
    model = mujoco.MjModel.from_xml_string(xml, _mesh_assets())
    model.opt.timestep = PHYS_DT
    model.opt.density = AIR_DENSITY   # enables the fluid model opt.wind acts through
    box_ids = [model.body(f"{BOX_PREFIX}{i}").id for i in range(len(boxes))]
    assert len(box_ids) == N_BOXES, f"expected {N_BOXES} box bodies, compiled {len(box_ids)}"
    _MODEL_CACHE[round_seed] = (model, box_ids)
    return model, box_ids


class ParkourSim:
    """Named to match the upstream base class the referee/history/preview call sites share;
    kept as ParkourSim rather than renamed so this fork's diff against upstream stays legible."""

    def __init__(self, params: InstanceParams):
        self.params = params
        self.model, self._box_ids = _round_scene(params.round_seed)
        self.model.opt.wind[:] = self.params.wind
        self.data = mujoco.MjData(self.model)
        self._pelvis = self.model.body("pelvis").id
        # Rays must hit the room (floor + boxes), not the robot. mj_ray filters by RENDER GROUP;
        # the room and every box are emitted into WORLD_GROUP (env/course.py), and the robot's own
        # geoms live in groups 0/1, so a mask admitting only WORLD_GROUP can never see the robot's
        # own legs as "the ground". There is no separate overhead-structure group in this course
        # (no duck-under) so the up-ray uses the same mask as the down-ray.
        self._ray_mask = np.zeros(6, np.uint8)
        self._ray_mask[WORLD_GROUP] = 1
        self._geomid = np.zeros(1, np.int32)
        self.steps = 0
        self.max_x = START_X
        self._action = np.zeros(ACT_DIM)
        self._seed = params.seed

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng([self._seed, 0xBADA55])
        self.data.qpos[0] = START_X
        self.data.qpos[2] = PLINTH_TOP + 0.793
        self.data.qpos[7:19] = DEFAULT_ANGLES
        # Robot qpos is [0:19] (free joint 7 + 12 leg angles); box free joints follow at [19:].
        # mj_resetData already sets every box to its compiled resting pose (qpos0), so nothing
        # box-related needs setting here -- only the robot's own noise is added, and only over
        # the robot's own qpos/qvel slice, so reset noise never nudges a box off its rest pose.
        self.data.qpos[:19] += rng.uniform(-RESET_NOISE, RESET_NOISE, 19)
        self.data.qvel[:18] += rng.uniform(-RESET_NOISE, RESET_NOISE, 18)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.max_x = START_X
        self._action = np.zeros(ACT_DIM)
        return self._obs()

    def step(self, action, max_steps: int = DEFAULT_MAX_STEPS) -> StepResult:
        a = np.asarray(action, dtype=np.float64).ravel()
        if a.shape != (ACT_DIM,):
            raise InvalidAction(f"action must be {ACT_DIM} floats, got shape {a.shape}")
        if not np.all(np.isfinite(a)):
            bad = int(np.count_nonzero(~np.isfinite(a)))
            raise InvalidAction(f"action must be finite; {bad} of {ACT_DIM} entries are NaN/inf")
        self._action = np.clip(a, -10.0, 10.0)
        target = self._action * ACTION_SCALE + DEFAULT_ANGLES
        for _ in range(FRAME_SKIP):
            self.data.ctrl[:] = (target - self.data.qpos[7:19]) * KP - self.data.qvel[6:18] * KD
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        self.max_x = max(self.max_x, float(self.data.qpos[0]))
        return StepResult(obs=self._obs(), terminal_reason=self._terminal(max_steps))

    @property
    def progress(self) -> float:
        """Fraction of the room crossed, in [0, 1]."""
        return float(np.clip((self.max_x - START_X) / (ROOM_LENGTH - START_X), 0.0, 1.0))

    # -- perception ------------------------------------------------------------------------

    def _yaw(self) -> float:
        qw, qx, qy, qz = self.data.qpos[3:7]
        return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))

    def _ray_down(self, x: float, y: float, z_from: float) -> float:
        """World height of the first surface below (x, y) -- floor OR a box top, whichever is
        higher -- or -SCAN_CLIP if there is none."""
        d = mujoco.mj_ray(self.model, self.data, np.array([x, y, z_from]),
                          np.array([0.0, 0.0, -1.0]), self._ray_mask, 1, -1, self._geomid)
        return z_from - d if d >= 0 else -SCAN_CLIP

    def _ray_up(self, x: float, y: float, z_from: float) -> float:
        """Clearance above (x, y, z_from) up to SCAN_CLIP; SCAN_CLIP if nothing overhead. Only
        boxes can occlude this course (no gantries/bars), so this reports how tall a box directly
        ahead is -- useful for telling "low crate, step over" from "climb-zone stack, mount it"
        before the height scan's forward reach gets there."""
        d = mujoco.mj_ray(self.model, self.data, np.array([x, y, z_from]),
                          np.array([0.0, 0.0, 1.0]), self._ray_mask, 1, -1, self._geomid)
        return SCAN_CLIP if d < 0 else min(d, SCAN_CLIP)

    def _obs(self) -> np.ndarray:
        d, yaw = self.data, self._yaw()
        px, py, pz = (float(v) for v in d.qpos[:3])
        c, s = np.cos(yaw), np.sin(yaw)

        rot = np.array(d.xmat[self._pelvis]).reshape(3, 3)
        lin = rot.T @ d.qvel[:3]
        ang = rot.T @ d.qvel[3:6]
        grav = rot.T @ np.array([0.0, 0.0, -1.0])

        scan = np.empty(SCAN_NX * SCAN_NY)
        k = 0
        for dx in SCAN_X:
            for dy in SCAN_Y:
                wx, wy = px + c * dx - s * dy, py + s * dx + c * dy
                scan[k] = self._ray_down(wx, wy, pz + RAY_FROM_ABOVE) - pz
                k += 1
        np.clip(scan, -SCAN_CLIP, SCAN_CLIP, out=scan)

        over = np.array([self._ray_up(px + c * dx, py + s * dx, pz + 0.05) for dx in OVERHEAD_X])
        ground = self._ray_down(px, py, pz + RAY_FROM_ABOVE)
        phase = (self.steps * PHYS_DT * FRAME_SKIP % GAIT_PERIOD) / GAIT_PERIOD

        return np.concatenate([
            grav,
            ang * 0.25,
            lin * 2.0,
            d.qpos[7:19] - DEFAULT_ANGLES,
            d.qvel[6:18] * 0.05,
            self._action,
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
            [np.sin(yaw), np.cos(yaw)],
            [py, (ROOM_LENGTH - px) / 10.0, np.clip(pz - ground, -SCAN_CLIP, SCAN_CLIP)],
            scan,
            over,
        ]).astype(np.float32)

    # -- gates -----------------------------------------------------------------------------

    def _terminal(self, max_steps: int) -> str | None:
        qpos, qvel = self.data.qpos, self.data.qvel
        if not (np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))):
            return "physics_glitch"
        if np.max(np.abs(qvel)) > QVEL_GLITCH_LIMIT:
            return "physics_glitch"
        if qpos[0] >= ROOM_LENGTH:
            return "completed"
        if abs(qpos[1]) > TRACK_HALF_W:
            return "out_of_bounds"
        px, py, pz = (float(v) for v in qpos[:3])
        if float(self.data.xmat[self._pelvis].reshape(3, 3)[2, 2]) < UPRIGHT_MIN:
            return "fell"
        if pz - self._ray_down(px, py, pz + RAY_FROM_ABOVE) < FALL_CLEARANCE:
            return "fell"
        if self.steps >= max_steps:
            return "timeout"
        return None


def _mesh_assets() -> dict[str, bytes]:
    """MuJoCo resolves meshdir relative to the XML's own path, which from_xml_string does not
    have. Hand it the STLs directly."""
    return {p.name: p.read_bytes() for p in (ASSETS / "meshes").glob("*.STL")}

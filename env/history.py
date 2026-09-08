"""One JSON file per evaluation instance: conditions, outcome, and the run itself.

Fork of Humanoid Parkour's history.py. The referee writes these to `/data/history/`, collected
by the platform as `FileType.HISTORY`. A directory is used because the suite runs all instances
in one container, so the per-game unit is a file.

Per frame: `qpos` for replay, and `action` — what the policy returned — as diagnostics.
Re-simulating from actions would depend on bit-exact physics and could silently drift; positions
cannot. Arrays are base64 float32.

`conditions` carries the box field the instance was run against -- every box's zone, position,
half-extents, density and yaw -- and the wind it was run under. That is the state a replay needs
to rebuild the scene, and it is what the record is for: the conditions the instance faced, not
the inputs that produced them. Neither the round seed nor the per-instance seed is recorded, the
same choice upstream parkour makes for its own per-geom friction array.

Lives in `env/` because the referee image copies it, not `tools/` — one format serves both.
"""

from __future__ import annotations

import base64
import json
import math
import pathlib
from typing import Any

import numpy as np

from .sim import FRAME_SKIP, PHYS_DT

# Bump the minor half when adding keys a reader can ignore, the major half when it cannot.
# box_scramble/1 replaced the inherited humanoid_parkour_history/1: `conditions` now carries the
# box field (see the module docstring), which a replay cannot ignore -- it is the scene.
#
# /2 (2026-09-08) splits `frames.qpos` into the robot's own qpos plus `frames.boxes`, which
# carries ONLY the boxes that actually moved. /1 wrote the whole model's qpos every frame -- 1422
# floats, of which 1372 were 196 box free joints that mostly never move. Measured over the PR-env
# load test (apex-mvp#452): 4.9 GB of history for 55 submissions, ~119 MB per submission, single
# instance files up to 11 MB, against parkour's ~50 KB. In a typical instance 30 of 199 boxes move
# more than a millimetre and 6 move more than a centimetre, so the robot plus the movers is ~17%
# of what /1 wrote. A reader fills every box in from `frames.boxes.rest` and animates the indices
# in `frames.boxes` over it -- see reconstruct_qpos(), which does exactly that.
FORMAT = "box_scramble_history/2"

# A box counts as moved if its centre shifts at least this far, in metres, at any recorded frame.
# Well below the smallest box half-extent, so a box that is genuinely nudged is always kept and
# solver jitter on a resting stack is not.
MOVED_EPS = 1e-3

# Fields of one recorded box, in this order, packed as a float32 (N_BOXES, 8) array. Zones are
# stored separately as a plain list of strings: they are labels, not numbers.
BOX_FIELDS = ("cx", "cy", "cz", "hx", "hy", "hz", "density", "yaw")

# Recording every control step is 50 Hz. 2 is the default because the replay renders at 25 fps
# anyway (tools/replay.py picks a stride to hit ~30 fps), so at stride 2 the video is IDENTICAL
# and the artifact is half the size. Set 1 when you want slow-motion frame-stepping.
DEFAULT_STRIDE = 2

# Only these appear in our own files; refuse anything else rather than hand an arbitrary dtype
# string from a downloaded artifact to numpy.
_ALLOWED_DTYPES = {"float32", "int32"}


def filename(index: int) -> str:
    return f"instance_{index:02d}.json"


def pack(a: np.ndarray) -> dict[str, Any]:
    """A numpy array as a JSON-safe {dtype, shape, b64} object."""
    a = np.ascontiguousarray(a)
    if a.dtype.name not in _ALLOWED_DTYPES:
        raise ValueError(f"refusing to pack dtype {a.dtype.name}")
    return {"dtype": a.dtype.name, "shape": list(a.shape),
            "b64": base64.b64encode(a.tobytes()).decode("ascii")}


def unpack(d: dict[str, Any]) -> np.ndarray:
    dtype = str(d["dtype"])
    if dtype not in _ALLOWED_DTYPES:
        raise ValueError(f"unexpected dtype {dtype!r} in history file")
    raw = base64.b64decode(d["b64"])
    return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape([int(x) for x in d["shape"]])


def reconstruct_qpos(record: dict[str, Any]) -> np.ndarray:
    """The full model qpos per frame, (frames, nq_model), as MuJoCo would set it.

    /2 records the robot's qpos plus only the boxes that moved, so the scene has to be rebuilt:
    every box starts at its `frames.boxes.rest` pose and stays there unless it appears in
    `frames.boxes.indices`. /1 files carried the whole model's qpos and are returned unchanged.

    Exact for the robot and for every box that moved; a box that did not move is placed at its
    frame-0 pose, so its error is its own drift and is below MOVED_EPS by construction.
    """
    frames = record["frames"]
    qpos = unpack(frames["qpos"])
    moved = frames.get("boxes")
    if moved is None:                      # box_scramble_history/1
        return qpos

    rest = unpack(moved["rest"])
    n_frames, nq_robot = qpos.shape
    full = np.zeros((n_frames, nq_robot + 7 * len(rest)), np.float32)
    full[:, :nq_robot] = qpos
    full[:, nq_robot:] = np.tile(rest.reshape(1, -1), (n_frames, 1))
    idx = [int(i) for i in moved["indices"]]
    if idx:
        poses = unpack(moved["qpos"])
        if poses.shape != (n_frames, len(idx), 7):
            raise ValueError(f"frames.boxes.qpos {poses.shape} does not match "
                             f"{n_frames} frames x {len(idx)} moved boxes")
        for k, i in enumerate(idx):
            j = nq_robot + 7 * i
            full[:, j:j + 7] = poses[:, k, :]
    return full


def pack_boxes(boxes: list[Any]) -> dict[str, Any]:
    """The round's box field as {zones, values} -- one row per box, columns `BOX_FIELDS`."""
    values = np.asarray([[getattr(b, f) for f in BOX_FIELDS] for b in boxes], np.float32)
    return {"fields": list(BOX_FIELDS), "zones": [str(b.zone) for b in boxes],
            "values": pack(values)}


def boxes_from_record(record: dict[str, Any]) -> list[Any]:
    """Rebuild the `env.course.Box` list a record was run against.

    Imported lazily so `tools/replay.py` can read a record without pulling in the course module's
    sampling code, and so this module stays importable in the referee image on its own.
    """
    from .course import Box

    field = record["conditions"]["box_field"]
    fields = [str(f) for f in field["fields"]]
    if fields != list(BOX_FIELDS):
        raise ValueError(f"unexpected box_field columns {fields}")
    values = unpack(field["values"])
    zones = [str(z) for z in field["zones"]]
    if values.shape != (len(zones), len(BOX_FIELDS)):
        raise ValueError(f"box_field values {values.shape} do not match {len(zones)} zones")
    return [Box(zone, *(float(v) for v in row)) for zone, row in zip(zones, values)]


class InstanceRecorder:
    """Buffers one instance's motion, then renders it as a history record.

        rec = InstanceRecorder(index, sim, stride)   # after sim.reset(), keeps the start pose
        rec.capture(sim, action)                      # after every sim.step()
        record = rec.record(sim, row, match_id=...)   # once the instance has terminated

    One instance at a time: the referee writes each file as its instance ends and drops the
    buffer, so peak memory is one episode (~0.5 MB) rather than the whole suite.
    """

    def __init__(self, index: int, sim, stride: int = DEFAULT_STRIDE):
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.index = int(index)
        self.stride = int(stride)
        self._qpos: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._ticks: list[int] = []
        self._nq = int(sim.model.nq)
        # The robot is the first body, so its qpos is the leading slice and the box free joints
        # (7 each) follow. Derived from the model rather than hardcoded, so this keeps working if
        # the robot's DoF count changes again.
        self._nq_robot = int(sim.model.nq) - 7 * len(sim.boxes)
        self._params = sim.params
        # The field this instance ran against, captured once: it is fixed for the whole round, so
        # recording it per instance costs ~6 KB and makes every file independently replayable.
        self._box_field = pack_boxes(sim.boxes)
        # Frame 0 is the starting pose, before any action has been taken.
        self._append(sim, None)

    def capture(self, sim, action) -> None:
        if sim.steps % self.stride == 0:
            self._append(sim, action)

    def record(self, sim, row: dict[str, Any], match_id: str | None = None,
               num_instances: int | None = None) -> dict[str, Any]:
        """Close the instance and return its JSON-ready record.

        The terminal pose is always kept even when it does not fall on the stride: the frame
        where the robot fails is the one worth looking at.
        """
        if not self._ticks or self._ticks[-1] != sim.steps:
            self._append(sim, None)
        return {
            "format": FORMAT,
            "match_id": match_id,
            "instance": self.index,
            "num_instances": num_instances,
            "conditions": {
                # Read off the instance's own params and its own compiled field, not the caller's
                # row, so the referee and the local tools cannot drift. Seeds are not conditions
                # and are not recorded (module docstring); the box field below is the geometry
                # this instance was actually run against, which is what a replay needs.
                "box_field": self._box_field,
                "wind_speed_ms": round(self._params.wind_speed, 2),
                "wind_dir_deg": round(math.degrees(self._params.wind_dir), 1),
            },
            "outcome": {k: row.get(k) for k in
                        ("terminal_reason", "progress", "distance_m", "steps", "sim_time_s",
                         "score") if k in row},
            "timing": {"phys_dt": PHYS_DT, "frame_skip": FRAME_SKIP,
                       "control_dt": PHYS_DT * FRAME_SKIP, "stride": self.stride},
            "frames": {
                "count": len(self._ticks),
                # The ROBOT's qpos width, not the model's -- see `nq_model` below and FORMAT.
                "nq": self._nq_robot,
                "nq_model": self._nq,
                # Control step each frame was captured at; frame 0 is the pre-step pose.
                "ticks": pack(np.asarray(self._ticks, np.int32)),
                # Position only. No qvel, so a replay can show the motion but not contact forces,
                # which need the full state to recompute. Add it here if that changes.
                "qpos": pack(np.asarray(self._qpos, np.float32)[:, :self._nq_robot]),
                "boxes": self._moved_boxes(),
                # Aligned with ticks. Frame 0 and the terminal frame carry zeros — no action
                # produced them.
                "action": pack(np.asarray(self._action, np.float32)),
            },
            "mujoco_version": _mujoco_version(),
        }

    def _moved_boxes(self) -> dict[str, Any]:
        """Per-frame pose for the boxes that moved, and nothing for the ones that did not.

        Returns {rest, indices, qpos}: `rest` is every box's (n_boxes, 7) pose at frame 0,
        `indices` are box numbers into it, and `qpos` is (frames, len(indices), 7) free-joint
        poses aligned with `frames.ticks`. A box absent from `indices` never moved: render it at
        its `rest` pose for the whole run.

        `rest` is recorded rather than derived from `conditions.box_field` because the two are not
        the same number: `boxes_xml_fragment` writes positions with %.3f, so what MuJoCo compiled
        is the box field rounded to a millimetre. Deriving the rest pose from the field therefore
        put a systematic sub-millimetre offset on every un-recorded box, on top of that box's own
        drift. Recording frame 0 costs ~5.6 KB and makes the reconstruction exact for every box
        that did not move, so the only error left is bounded by MOVED_EPS by construction.
        """
        frames = np.asarray(self._qpos, np.float32)
        boxes = frames[:, self._nq_robot:]
        n_boxes = boxes.shape[1] // 7
        if n_boxes == 0:
            return {"rest": pack(np.zeros((0, 7), np.float32)), "indices": [],
                    "qpos": pack(np.zeros((len(frames), 0, 7), np.float32))}
        boxes = boxes.reshape(len(frames), n_boxes, 7)
        shift = np.linalg.norm(boxes[:, :, :3] - boxes[0:1, :, :3], axis=2).max(axis=0)
        idx = np.flatnonzero(shift >= MOVED_EPS)
        return {"rest": pack(np.ascontiguousarray(boxes[0])),
                "indices": [int(i) for i in idx],
                "qpos": pack(np.ascontiguousarray(boxes[:, idx, :]))}

    def _append(self, sim, action) -> None:
        self._qpos.append(np.asarray(sim.data.qpos, np.float32))
        self._ticks.append(int(sim.steps))
        if action is None:
            self._action.append(np.zeros(len(sim.data.ctrl), np.float32))
        else:
            a = np.asarray(action, np.float32).ravel()
            self._action.append(a if a.shape == (len(sim.data.ctrl),)
                                else np.zeros(len(sim.data.ctrl), np.float32))


def write_instance(directory: str | pathlib.Path, record: dict[str, Any]) -> pathlib.Path:
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename(int(record["instance"]))
    path.write_text(json.dumps(record))
    return path


def read_instance(path: str | pathlib.Path) -> dict[str, Any]:
    record = json.loads(pathlib.Path(path).read_text())
    got = str(record.get("format", "?"))
    if got.split("/")[0] != FORMAT.split("/")[0]:
        raise ValueError(f"{path} is not an evaluation history file (format {got!r})")
    return record


def read_all(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """Every history record at `path` — a directory of them, or one file."""
    path = pathlib.Path(path)
    if path.is_dir():
        found = sorted(path.glob("*instance_*.json"))
        if not found:
            raise FileNotFoundError(f"no instance_NN.json files in {path}")
        return [read_instance(p) for p in found]
    return [read_instance(path)]


def _mujoco_version() -> str:
    """Provenance only — replay reads positions, so it is not version-locked."""
    try:
        import mujoco
        return str(mujoco.__version__)
    except Exception:
        return "unknown"

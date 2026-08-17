"""One JSON file per evaluation instance: conditions, outcome, and the run itself.

Fork of Humanoid Parkour's history.py. The referee writes these to `/data/history/`, collected
by the platform as `FileType.HISTORY`. A directory is used because the suite runs all instances
in one container, so the per-game unit is a file.

Per frame: `qpos` for replay, and `action` — what the policy returned — as diagnostics.
Re-simulating from actions would depend on bit-exact physics and could silently drift; positions
cannot. Arrays are base64 float32.

`conditions.round_seed` is stored (unlike upstream parkour, which deliberately omits its
equivalent) because this fork's box field is compile-time geometry keyed by round seed, not a
runtime friction value -- a replay has no way to reconstruct box sizes/positions/densities
without it, whereas parkour's fixed geometry needed only its per-geom friction array. This is
safe to include here for the same reason parkour omits the analogous field there: round seeds are
only revealed in metadata after a round completes, at which point the box field they produced is
no longer secret (the round is over), so recording it in a post-round-revealed history file leaks
nothing about a FUTURE round's field. See docs/design.md for why the room geometry (unlike
parkour's) is intentionally NOT public between rounds.

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
FORMAT = "humanoid_parkour_history/1"

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
        self._params = sim.params
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
                # Read off the instance's own params, not the caller's row, so the referee and
                # the local tools cannot drift. `params.seed` (the per-INSTANCE seed; reset noise
                # only) is deliberately NOT included, same as upstream parkour -- but
                # `round_seed` IS, because it is the only handle a replay has on this fork's
                # compile-time box geometry, and by the time a history file is revealed
                # (post-round) the round it describes is over. See the module docstring.
                "round_seed": self._params.round_seed,
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
                "nq": self._nq,
                # Control step each frame was captured at; frame 0 is the pre-step pose.
                "ticks": pack(np.asarray(self._ticks, np.int32)),
                # Position only. No qvel, so a replay can show the motion but not contact forces,
                # which need the full state to recompute. Add it here if that changes.
                "qpos": pack(np.asarray(self._qpos, np.float32)),
                # Aligned with ticks. Frame 0 and the terminal frame carry zeros — no action
                # produced them.
                "action": pack(np.asarray(self._action, np.float32)),
            },
            "mujoco_version": _mujoco_version(),
        }

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

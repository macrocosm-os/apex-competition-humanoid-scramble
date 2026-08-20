"""A recorded run must still replay from its history files alone.

History is only worth writing if it can be read back, and the writer (env/history.py, shared with
the referee) and the reader (tools/replay.py) are different files — so assert the round trip, not
just that recording ran. This course's records carry the box field they ran against rather than
the seed that drew it, so the replay has to rebuild the scene from the record's own conditions.

    PYTHONPATH=. python .github/scripts/check_replay_roundtrip.py <history-dir> <n>
"""

from __future__ import annotations

import json
import pathlib
import sys

import mujoco
import numpy as np

from env.history import read_all
from env.sim import ACT_DIM
from tools.preview import _lit_model_for_boxes
from tools.replay import Run

directory = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])

# One file per instance, named as the platform's FileType.HISTORY collector expects.
names = sorted(p.name for p in directory.glob("*.json"))
assert names == [f"instance_{i:02d}.json" for i in range(expected)], names

records = read_all(directory)
runs = [Run(r) for r in records]
assert len(runs) == expected, len(runs)

for record, run in zip(records, runs):
    # A record is revealed to the miner when the round closes. It carries the conditions the
    # instance faced; it must not carry the inputs that produced them.
    blob = json.dumps(record)
    assert "round_seed" not in blob, "a record carries a seed"
    assert "seed" not in record["conditions"], record["conditions"].keys()

    # The history has to reproduce the quantity the run was scored on. Recorded frames are a
    # subsample (stride) plus the terminal frame, so their max can only fall short of the true
    # max, never exceed it.
    recorded_max = float(np.nanmax(run.qpos[:, 0]))
    scored_max = float(run.outcome["distance_m"])
    assert recorded_max <= scored_max + 0.01, (recorded_max, scored_max)
    assert scored_max - recorded_max < 0.5, (recorded_max, scored_max)
    assert run.action.shape == (run.frames, ACT_DIM), run.action.shape

    # ...and rebuild the scene and the pose with nothing but mujoco and the record. A run that
    # ended in physics_glitch can genuinely have left the finite regime, so replay the last
    # finite frame rather than asserting the terminal one is finite: what is being checked is
    # that the record reconstructs a scene and a pose, not that every run stayed well-behaved.
    finite = np.flatnonzero(np.isfinite(run.qpos).all(axis=1))
    assert finite.size, f"instance {run.index} recorded no finite frame"
    model = _lit_model_for_boxes(run.boxes)
    data = mujoco.MjData(model)
    data.qpos[:] = run.qpos[finite[-1]]
    mujoco.mj_forward(model, data)
    assert np.isfinite(data.body("pelvis").xpos).all()

print(f"ok: replayed {sum(r.frames for r in runs)} frames from qpos and the recorded box field")

"""Export the room's STATIC geometry as renderer-friendly JSON, for the dashboard's course
preview.

Run from the repository root:

    PYTHONPATH=. python tools/export_course_json.py

    WHAT IS AND IS NOT IN HERE, because this competition differs from its siblings on exactly
    this point. Humanoid Parkour and Humanoid Olympics have static courses, so one exported file
    describes everything a renderer needs. Box Scramble does not: the 196-box field is drawn per
    ROUND from the platform's seed (env/course.sample_boxes), so no file checked into this repo
    can describe the boxes a given round was run against.

    So this file is the ROOM SHELL only -- floor slabs, walls, the elevated finish platform, the
    fixed leap chain, and the zone bands. It is correct for every round and never needs
    regenerating unless the room's own dimensions change.

    THE BOXES COME FROM THE HISTORY RECORD. Every per-instance history file carries
    `conditions.box_field` (every box's zone, centre, half-extents, density and yaw) and
    `frames.boxes` (the rest pose of all of them, plus per-frame poses for the ones that moved) --
    see env/history.py. A submission viewer therefore renders: this shell, then the round's boxes
    from the record, then the robot. A competition PREVIEW with no submission to draw from can
    only show the shell plus, if it wants representative clutter, a field sampled from any seed.

Sibling of tools/export_course_json.py in apex-competition-humanoid-olympics; same surface shape
(`center_m` / `size_m` / `yaw_rad` / `kind` / `rgba`) so one renderer can read both.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from env.course import (APRON_LEN, CLIMB_X0, CLIMB_LEN, COLOR, DASH_LEN, FIELD_LEN, FINISH_RISE,
                        N_CLIMB, N_PUSH, N_SCRAMBLE, PLINTH_TOP, PUSH_X0, PUSH_LEN, ROOM_LENGTH,
                        SCRAMBLE_X0, SCRAMBLE_LEN, TRACK_HALF_W, build_floor,
                        _leap_chain_boxes)
from env.sim import START_X


def _surface(box, kind: str) -> dict:
    """One static slab, as full extents and a world centre."""
    cx, cy, cz, hx, hy, hz, ck = box
    return {"shape": "box", "kind": ck, "supports_robot": True,
            "center_m": [round(cx, 4), round(cy, 4), round(cz, 4)],
            "size_m": [round(2 * hx, 4), round(2 * hy, 4), round(2 * hz, 4)],
            "yaw_rad": 0.0,
            "rgba": [float(v) for v in COLOR[ck].split()]}


def _leap_json() -> list[dict]:
    """The fixed leap chain: part of the room, not of the sampled field."""
    out = []
    for b in _leap_chain_boxes():
        # Tagged `leap` rather than `climb`: it shares the climb palette but it is fixed room
        # furniture, not sampled field, and a renderer wants to tell those apart.
        out.append({"shape": "box", "kind": "leap", "supports_robot": True,
                    "center_m": [round(b.cx, 4), round(b.cy, 4), round(b.cz, 4)],
                    "size_m": [round(2 * b.hx, 4), round(2 * b.hy, 4), round(2 * b.hz, 4)],
                    "yaw_rad": round(float(b.yaw), 4),
                    "rgba": [float(v) for v in COLOR[b.zone].split()]})
    return out


def export() -> dict:
    surfaces = []
    for seg in build_floor():
        for box in seg.boxes:
            surfaces.append(_surface(box, seg.kind))
    return {
        "schema": "box-scramble.course-layout.v1",
        "units": "metres and radians",
        "coordinate_system": {
            "x": "along the room, start to finish",
            "y": "robot-left, 0 on the centreline",
            "z": "up",
            "deck_top_z_m": PLINTH_TOP,
        },
        "room": {
            "length_m": ROOM_LENGTH,
            "full_width_m": 2 * TRACK_HALF_W,
            "half_width_m": TRACK_HALF_W,
            "apron_len_m": APRON_LEN,
            "field_len_m": FIELD_LEN,
            "dash_len_m": DASH_LEN,
        },
        "start": {"position_m": [START_X, 0.0, PLINTH_TOP], "yaw_rad": 0.0},
        "finish": {
            # Completion needs BOTH: past the far wall AND up on the platform. A renderer that
            # draws only the x line will mislead, since staying on the floor never completes.
            "x_m": ROOM_LENGTH,
            "platform_top_z_m": round(PLINTH_TOP + FINISH_RISE, 4),
            "rise_above_deck_m": FINISH_RISE,
        },
        "zones": [
            {"id": "scramble", "x0_m": SCRAMBLE_X0, "length_m": SCRAMBLE_LEN, "boxes": N_SCRAMBLE,
             "rgba": [float(v) for v in COLOR["scramble"].split()]},
            {"id": "push", "x0_m": PUSH_X0, "length_m": PUSH_LEN, "boxes": N_PUSH,
             "rgba": [float(v) for v in COLOR["push"].split()]},
            {"id": "climb", "x0_m": CLIMB_X0, "length_m": CLIMB_LEN, "boxes": N_CLIMB,
             "rgba": [float(v) for v in COLOR["climb"].split()]},
        ],
        "surfaces": surfaces + _leap_json(),
        # Stated in the payload, not just this module's docstring, so a renderer reading only the
        # file knows the shell is not the whole scene.
        "box_field": {
            "static": False,
            "sampled_per_round": True,
            "count": N_SCRAMBLE + N_PUSH + N_CLIMB,
            "source": "per-instance history record: conditions.box_field and frames.boxes",
            "note": "Drawn from the round seed inside the referee. Not derivable from this file.",
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/course-layout.json"))
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(export(), indent=2) + "\n")
    print(f"{a.out} ({a.out.stat().st_size / 1000:.1f} kB)")

"""The room's and the crates' friction must reach the solver, not just the model.

MuJoCo mixes contact friction as the element-wise MAXIMUM at equal `geom_priority`, and
`g1_22dof.xml` declares none, so the robot's feet default to 1.0 and discard every mu this course
draws. Inert in 0.1.0-0.1.2; same defect and same fix as humanoid_parkour v0.7.0 and
humanoid_olympics 0.2.0.

Asserts on SOLVED contact friction: a score cannot tell an applied band from one mixed away.

    PYTHONPATH=. python .github/scripts/check_friction_reaches_contacts.py
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

from env import ParkourSim, instance_spec
from env.course import BOX_PREFIX, GEOM_PREFIX
from env.sim import ACT_DIM

SETTLE_STEPS = 60
TOL = 1e-4
SEED = 1885531765


def geom_name(model, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""


def is_surface(model, gid: int) -> bool:
    name = geom_name(model, gid)
    return name.startswith(GEOM_PREFIX) or name.startswith(BOX_PREFIX)


def settle(sim: ParkourSim, steps: int = SETTLE_STEPS) -> ParkourSim:
    for _ in range(steps):
        sim.step(np.zeros(ACT_DIM), max_steps=10_000)
    return sim


def surface_contacts(sim: ParkourSim) -> list[tuple[str, float, float]]:
    """(geom name, declared mu, solved mu) for ROBOT-versus-surface contacts.

    Surface/surface pairs are excluded: a 0.75 crate on the 0.90 floor correctly mixes to 0.90.
    The invariant is only about a surface losing to the robot's 1.0 default.
    """
    out = []
    for c in range(sim.data.ncon):
        con = sim.data.contact[c]
        pair = (con.geom1, con.geom2)
        surfaces = [g for g in pair if is_surface(sim.model, g)]
        if len(surfaces) != 1:          # skip surface/surface and robot/robot
            continue
        gid = surfaces[0]
        out.append((geom_name(sim.model, gid),
                    float(sim.model.geom_friction[gid, 0]),
                    float(con.friction[0])))
    return out


failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"ok: {label}")
    else:
        failures.append(f"{label}: {detail}")
        print(f"FAIL: {label}: {detail}")


sim = ParkourSim(instance_spec(0, 24, SEED))
sim.reset()
model = sim.model

# Surfaces must outrank the robot, or max() discards whatever they asked for.
surface_prio, other_prio = set(), set()
for gid in range(model.ngeom):
    (surface_prio if is_surface(model, gid) else other_prio).add(int(model.geom_priority[gid]))
check("room and crate geoms outrank the robot",
      bool(surface_prio) and min(surface_prio) > max(other_prio),
      f"surfaces {sorted(surface_prio)} vs everything else {sorted(other_prio)}")

# Crates must ask for sub-1.0 mu, or this cannot tell a working fix from max() picking the foot.
crate_mus = [float(model.geom_friction[g, 0]) for g in range(model.ngeom)
             if geom_name(model, g).startswith(BOX_PREFIX)]
check("crates ask for mu below the robot's 1.0 default",
      bool(crate_mus) and max(crate_mus) < 1.0 - TOL,
      f"{len(crate_mus)} crates, mu {min(crate_mus):.3f}-{max(crate_mus):.3f}" if crate_mus else "none")

# Density drives crate friction, so a field of one mu means the axis is decorative anyway.
check("crate friction varies across the field",
      len({round(m, 3) for m in crate_mus}) >= 10,
      f"{len({round(m, 3) for m in crate_mus})} distinct mu values across {len(crate_mus)} crates")

# The failure signature: declared well below 1.0, solved at exactly 1.0.
settle(sim)
contacts = surface_contacts(sim)
check("the robot forms contacts with the room", bool(contacts), "no surface contacts formed")
if contacts:
    clamped = [(n, d, s) for n, d, s in contacts if abs(s - d) > TOL]
    check("solved contact mu equals the declared mu",
          not clamped,
          "; ".join(f"{n} asked {d:.4f} got {s:.4f}" for n, d, s in clamped[:4]))
    sub_one = [(n, d, s) for n, d, s in contacts if d < 1.0 - TOL]
    check("a sub-1.0 surface is exercised, not just the mu-1.0 start platform",
          bool(sub_one),
          f"every contacted surface declared mu >= 1.0: {sorted({round(d,3) for _, d, _ in contacts})}")
    pinned = [(n, d, s) for n, d, s in sub_one if abs(s - 1.0) <= TOL]
    check("no sub-1.0 surface is pinned to MuJoCo's 1.0 default",
          not pinned,
          "; ".join(f"{n} asked {d:.4f} solved 1.0" for n, d, _ in pinned[:4]))

# Drop the robot onto a crate: the siblings' guards never had to cover this family.
crates = [(g, float(model.geom_friction[g, 0])) for g in range(model.ngeom)
          if geom_name(model, g).startswith(BOX_PREFIX)]
target_gid, target_mu = min(crates, key=lambda t: t[1])       # slipperiest crate in the field
on_crate = ParkourSim(instance_spec(0, 24, SEED))
on_crate.reset()
pos = on_crate.data.geom_xpos[target_gid].copy()
half_z = float(on_crate.model.geom_size[target_gid][2])
on_crate.data.qpos[0] = float(pos[0])
on_crate.data.qpos[1] = float(pos[1])
on_crate.data.qpos[2] = float(pos[2]) + half_z + 0.80        # pelvis above its top face
mujoco.mj_forward(on_crate.model, on_crate.data)
settle(on_crate, 120)
crate_contacts = [(n, d, s) for n, d, s in surface_contacts(on_crate) if n.startswith(BOX_PREFIX)]
check("a crate contact is exercised",
      bool(crate_contacts),
      f"robot never touched a crate after being placed on geom {target_gid} (mu {target_mu:.3f})")
if crate_contacts:
    bad = [(n, d, s) for n, d, s in crate_contacts if abs(s - d) > TOL]
    check("crate contacts solve at the crate's own mu",
          not bad,
          "; ".join(f"{n} asked {d:.4f} got {s:.4f}" for n, d, s in bad[:4]))

if failures:
    sys.exit("friction does not reach the solver:\n  " + "\n  ".join(failures))
print(f"\nok: friction reaches the solver -- {len(crate_mus)} crates spanning "
      f"mu {min(crate_mus):.3f}-{max(crate_mus):.3f}, all solving at their declared value")

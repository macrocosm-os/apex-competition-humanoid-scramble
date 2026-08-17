"""The Box Scramble course: one room, a fixed number of boxes, per-round sampled sizes/densities.

Fork of Humanoid Parkour's linear plinth course. Same robot (Unitree G1, legs-only), same
observation/action contract, same referee/player split — what changes is the obstacle: instead of
a fixed sequence of named maneuvers (ramp, stairs, hurdle, ...), the robot crosses a long room
scattered with loose boxes and has to dash across the clear lane, scramble over a dense cluster,
shove lightweight crates out of the only path, and climb a stack too tall to step onto directly.

ROOM GEOMETRY (why 12 m x 6 m, aspect 2:1):
    The brief asked for a 2:1 room. Width sets how much lateral routing choice exists (go left
    around a pile, right, or through) before boxes stop clustering into anything forced; length
    at 2x width is what turns "an obstacle" into "a room you cross". Numbers, not vibes:

    - Width 6 m gives three ~2 m lanes -- enough for a real sidestep detour around one box pile
      without being so wide a policy can always find empty floor and never has to interact with
      anything. TRACK_HALF_W below is 3.0 m (half of 6 m), vs. 1.2 m on the linear parkour course
      -- this course is a room, not a corridor, and the width has to say so.
    - Length 12 m = 2x width, per the brief. Split into a start apron, three obstacle zones, and
      a dash finish (below) -- long enough for each zone to read as a distinct phase of the
      crossing, short enough that a 50 Hz control loop finishes it well inside the wall-clock
      budget parkour was already sized against (see spec.yaml).

BOX COUNT (why 20, fixed):
    Floor area = 6 x 12 = 72 sq m. A course that is all dash lanes has nothing to scramble over;
    a course that is wall-to-wall boxes is a wall, not an obstacle -- the design goal (see
    docs/design.md) is congestion dense enough to force weaving/pushing/climbing in three zones,
    with real open floor in between and at the ends. Target packing fraction: ~20% of floor area
    covered by box footprints, non-uniformly distributed (clustered per zone, not gridded).

    Box footprint area is sampled per box (see BOX_SIZE_DIST below); its mean works out to
    ~0.72 sq m ((0.85 m avg side)^2). At 20% packing: 0.20 x 72 / 0.72 ~= 20 boxes. This is
    ONE FIXED COUNT for the whole course (the brief's "fixed number of boxes"), split
    deterministically across the three zones by role (9 scramble / 5 push / 6 climb) --
    what's sampled per round is each box's size and density, not how many there are or which
    zone it lives in.

ZONES, west to east along +x (see build_course):
    0.0 - 2.0 m   start apron        clear; let the policy settle into gait before any obstacle
    2.0 - 5.0 m   SCRAMBLE field     9 small/medium boxes, moderate density, tightly clustered:
                                     no straight-line path exists, only weaving between/over them
    5.0 - 8.0 m   PUSH corridor      5 low-density (light) boxes placed directly in the only
                                     viable lane -- shovable by body contact, or a tight detour
    8.0 - 11.0 m  CLIMB stack        6 high-density (heavy, stable, high-friction) boxes sized
                                     and placed to land as 2-3 stackable tiers spanning the lane
                                     -- too tall for a single-leg step-up, must be climbed
    11.0 - 12.0 m dash finish        clear straight sprint to the line

    This mirrors the linear parkour course's own principle (progress-based scoring needs a
    continuous difficulty gradient, not discrete tiers) while giving the room-crossing brief its
    four verbs: dash (start apron + finish), scramble (dense small clutter), push (light boxes
    in-path), climb (heavy stacked boxes).

SIZE / DENSITY DISTRIBUTIONS (sampled per box, per round, from the round seed):
    Every box is an axis-aligned MJCF box geom. Size and density are drawn per-instance from
    zone-conditioned distributions (BOX_SIZE_DIST, BOX_DENSITY_DIST) so the *shapes* of each pile
    vary round to round even though the box count and zone roles are fixed -- this is what keeps
    the course from being memorisable while still keeping the room's difficulty curve stable
    (same principle as parkour's per-instance friction/wind: geometry that matters is randomised
    per round, not per submission, and is not observable as a number -- see env/sim.py).

    Density maps directly to two things a legs-only robot without grasping actually feels:
    - PUSHABILITY: MuJoCo derives box mass from density x volume. A box light enough for a
      152 kg-equivalent body-check (see docs/design.md, "push corridor sizing") to displace
      measurably is "pushable"; the push zone's density band is calibrated to that.
    - CLIMBABILITY: a stack is only good footing if the top box doesn't slide/tip when stood on.
      Heavier boxes have proportionally higher friction (BOX_FRICTION_BY_DENSITY) and the climb
      zone's boxes are sized so stacking is geometrically favoured (broad, squat boxes, low CoM)
      -- see stack-packing notes below.

    python -m env.course        # print the layout for a given seed
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

PLINTH_TOP = 0.8             # top surface of the room floor/plinth
PLINTH_THICK = 0.4
TRACK_HALF_W = 3.0            # 6 m room width (aspect 2:1 against the 12 m length below)
ROOM_LENGTH = 12.0             # total crossing distance, 2x TRACK_HALF_W*2 (2:1 aspect)

WORLD_GROUP = 2                # course + floor geoms; see env/sim.py for the ray-cast contract
GEOM_PREFIX = "course_"        # static room/floor geoms (walkable, ray-visible)
BOX_PREFIX = "box_"            # sampled boxes (walkable AND potentially movable)

# Boxes are free bodies (joint="free"), not welded geoms, so MuJoCo's contact solver is what
# makes "push" and "climb" real physical behaviours rather than scripted ones: a light box
# accelerates under sustained body contact; a heavy box in a stack barely moves and a robot can
# stand on it. Both zones' colours below double as the size legend in tools/preview.py renders.
COLOR = {
    "apron": ".55 .57 .60 1", "dash": ".55 .57 .60 1",
    "scramble": ".85 .55 .20 1", "push": ".35 .70 .95 1", "climb": ".70 .25 .20 1",
}

# Zone extents along +x, in metres from the start line (x=0 is the start apron's near edge).
APRON_LEN = 2.0
SCRAMBLE_LEN = 3.0
PUSH_LEN = 3.0
CLIMB_LEN = 3.0
DASH_LEN = 1.0
assert APRON_LEN + SCRAMBLE_LEN + PUSH_LEN + CLIMB_LEN + DASH_LEN == ROOM_LENGTH

ZONE_X0 = {
    "scramble": APRON_LEN,
    "push": APRON_LEN + SCRAMBLE_LEN,
    "climb": APRON_LEN + SCRAMBLE_LEN + PUSH_LEN,
}
ZONE_LEN = {"scramble": SCRAMBLE_LEN, "push": PUSH_LEN, "climb": CLIMB_LEN}

# Fixed box count per zone -- the brief's "fixed number of boxes", split by role. Total 20,
# derived from the ~20% floor-packing target above. NOT sampled: this is deterministic so the
# course's difficulty shape is stable round to round; only each box's size/density/exact
# placement jitter is drawn from the round seed.
N_SCRAMBLE, N_PUSH, N_CLIMB = 9, 5, 6
N_BOXES = N_SCRAMBLE + N_PUSH + N_CLIMB   # 20

# Size distributions: (kind -> (lognormal median side length m, sigma, min, max)) per axis pair.
# Scramble: small-to-medium clutter, roughly cubic, sized so single boxes are step-overable
# (<=0.5 m) but a CLUSTER of them (see placement jitter) forces weaving rather than one clean
# stride through. Push: flatter, lower-profile crates -- low enough that shoving one is a
# push, not a climb. Climb: broad and squat by design (see _sample_climb_dims) so stacking is
# geometrically stable rather than a lucky topple.
SCRAMBLE_SIDE = (0.42, 0.28, 0.28, 0.85)      # median, sigma, min, max (metres, per half-extent x2)
SCRAMBLE_HEIGHT = (0.34, 0.30, 0.22, 0.60)
PUSH_SIDE = (0.55, 0.20, 0.35, 0.80)
PUSH_HEIGHT = (0.30, 0.20, 0.18, 0.42)
CLIMB_SIDE = (0.70, 0.18, 0.45, 1.00)
CLIMB_HEIGHT = (0.34, 0.15, 0.26, 0.46)        # per tier; stacked 2-3 tiers -> 0.6-1.3 m total

# Density bands, kg/m^3. Calibrated against what a ~32 kg legs-only G1 can plausibly move with a
# body-check vs. what it needs as stable footing (docs/design.md, "push corridor sizing" and
# "climb stack sizing"). Real-world anchors: dry foam/cardboard ~15-60, packed textiles/light
# plastics ~150-350, water ~1000, wet sand/dense-packed goods ~1400-1900, similar to a
# loaded packing crate.
DENSITY_SCRAMBLE = (40.0, 260.0)     # light clutter -- easy to knock, not the point of this zone
DENSITY_PUSH = (25.0, 90.0)          # deliberately LOW: must be shovable by contact force alone
DENSITY_CLIMB = (350.0, 1400.0)      # deliberately HIGH: stable footing, doesn't slide/tip

# Sliding friction scales with density (heavier, denser material grips better in this course's
# fiction — think rubberised crate vs. slick lightweight tote), same spirit as parkour's
# friction-band-by-surface-kind. Returns (lo, hi) mu band for a given density.
def _friction_band(density: float) -> tuple[float, float]:
    lo = float(np.interp(density, [20.0, 1400.0], [0.15, 0.55]))
    hi = float(np.interp(density, [20.0, 1400.0], [0.35, 0.95]))
    return lo, hi


@dataclass
class Box:
    zone: str
    cx: float
    cy: float
    cz: float           # centre height of resting position (tier stacking handled by caller)
    hx: float
    hy: float
    hz: float            # half-extents
    density: float
    yaw: float = 0.0      # small random yaw jitter so piles don't look gridded


@dataclass
class Seg:
    """A static (non-box) floor slab -- start apron, zone floors, dash finish. Kept as its own
    type (rather than folding into Box) because these are welded geoms, not free bodies: the
    room's floor does not move, only the boxes on it do."""
    kind: str
    length: float
    boxes: list = field(default_factory=list)   # (cx, cy, cz, hx, hy, hz, color)


def _slab(x0, length, top, color, half_w=TRACK_HALF_W):
    return (x0 + length / 2, 0.0, top - PLINTH_THICK / 2, length / 2, half_w, PLINTH_THICK / 2, color)


def build_floor() -> list[Seg]:
    """The room's static floor: one slab across the full width for the whole 12 m length, plus
    the start ramp-on apron. A single flat plinth top (no stairs, no ramps) is deliberate -- this
    course's difficulty is entirely the box field, not the floor, so the floor stays a constant
    the policy can rely on."""
    segs = []
    segs.append(Seg("apron", APRON_LEN, [_slab(0.0, APRON_LEN, PLINTH_TOP, "apron")]))
    segs.append(Seg("field", SCRAMBLE_LEN + PUSH_LEN + CLIMB_LEN,
                    [_slab(APRON_LEN, SCRAMBLE_LEN + PUSH_LEN + CLIMB_LEN, PLINTH_TOP, "apron")]))
    segs.append(Seg("dash", DASH_LEN,
                    [_slab(APRON_LEN + SCRAMBLE_LEN + PUSH_LEN + CLIMB_LEN, DASH_LEN,
                           PLINTH_TOP, "dash")]))
    return segs


def _lognormal_clip(rng: np.random.Generator, median: float, sigma: float, lo: float, hi: float) -> float:
    """Lognormal draw with a given median (not mean), clipped to [lo, hi]. Lognormal rather than
    uniform/normal because real clutter/crate size distributions are right-skewed -- lots of
    smallish boxes, a long thin tail of big ones -- and it can never draw a negative size."""
    val = float(rng.lognormal(mean=math.log(median), sigma=sigma))
    return float(np.clip(val, lo, hi))


def _sample_scramble_boxes(rng: np.random.Generator) -> list[Box]:
    """Small/medium clutter, densely clustered with sub-body-width gaps so there is no single
    clean stride through -- the robot has to weave between and step over/on to make progress.
    Placed in two roughly-parallel clusters offset across the lane width so a straight line
    from apron to push-corridor threads directly through boxes no matter which lane is chosen."""
    boxes = []
    x0 = ZONE_X0["scramble"]
    # Two staggered rows across the 6 m width, offset in x, so a robot going straight down any
    # of the three lanes still meets a box: gaps between row-1 boxes are covered by row-2 boxes.
    ys = np.linspace(-TRACK_HALF_W + 0.5, TRACK_HALF_W - 0.5, 5)
    xs_row = [x0 + 0.6, x0 + 1.5, x0 + 2.4]
    slots = [(x, y) for x in xs_row for y in ys]
    rng.shuffle(slots)
    for i in range(N_SCRAMBLE):
        sx, sy = slots[i % len(slots)]
        side = _lognormal_clip(rng, *SCRAMBLE_SIDE)
        h = _lognormal_clip(rng, *SCRAMBLE_HEIGHT)
        density = float(rng.uniform(*DENSITY_SCRAMBLE))
        jx, jy = rng.uniform(-0.35, 0.35, 2)
        yaw = float(rng.uniform(-0.4, 0.4))
        boxes.append(Box("scramble", sx + jx, sy + jy, PLINTH_TOP + h / 2,
                        side / 2, side / 2, h / 2, density, yaw))
    return boxes


def _sample_push_boxes(rng: np.random.Generator) -> list[Box]:
    """Low-density boxes placed IN the lane, one per lane-width slice, so displacing one is the
    intended solution: the only alternative is squeezing past in a track that is deliberately
    narrower than a G1's shoulder-width clearance once a box sits at lane centre."""
    boxes = []
    x0 = ZONE_X0["push"]
    xs = np.linspace(x0 + 0.5, x0 + PUSH_LEN - 0.5, N_PUSH)
    for i, sx in enumerate(xs):
        # Alternate/jitter y so consecutive boxes don't line up into one wall to walk around;
        # each is still close enough to TRACK centre that skirting it means hugging a side wall.
        sy = float(rng.uniform(-0.6, 0.6)) + (0.0 if i % 2 == 0 else float(rng.choice([-1, 1])) * 0.5)
        sy = float(np.clip(sy, -TRACK_HALF_W + 0.6, TRACK_HALF_W - 0.6))
        side = _lognormal_clip(rng, *PUSH_SIDE)
        h = _lognormal_clip(rng, *PUSH_HEIGHT)
        density = float(rng.uniform(*DENSITY_PUSH))
        boxes.append(Box("push", sx, sy, PLINTH_TOP + h / 2, side / 2, side / 2, h / 2, density))
    return boxes


def _sample_climb_boxes(rng: np.random.Generator) -> list[Box]:
    """Heavy, broad, squat boxes stacked into 2-3 tiers spanning the lane -- too tall to step
    onto in one motion (tier tops land at 0.6-1.3 m, vs. parkour's proven-feasible 0.55 m single
    step), so a real climb (mount tier 1, stand, mount tier 2, ...) is the only way across.

    Stack-packing: each stack's footprint shrinks going up (base tier widest) so the pile is
    self-stable under gravity without scripted joints -- the same reason a real box stack is
    pyramided. Two stacks side by side leave a narrow gap between them as an alternate,
    harder-to-balance route for a policy that would rather thread the middle than climb either."""
    boxes = []
    x0 = ZONE_X0["climb"]
    stack_centers = [(x0 + 1.2, -1.4), (x0 + 1.2, 1.4), (x0 + 2.1, 0.0)]
    per_stack = [0, 0, 0]
    order = list(range(N_CLIMB))
    rng.shuffle(order)
    for idx, i in enumerate(order):
        stack_i = idx % len(stack_centers)
        tier = per_stack[stack_i]
        per_stack[stack_i] += 1
        cx0, cy0 = stack_centers[stack_i]
        side = _lognormal_clip(rng, *CLIMB_SIDE) * (1.0 - 0.12 * tier)   # narrower per tier up
        h = _lognormal_clip(rng, *CLIMB_HEIGHT)
        density = float(rng.uniform(*DENSITY_CLIMB))
        # z is resolved properly in build_course once tier heights for this stack are known;
        # placeholder height carries the tier index for the second pass.
        jx, jy = rng.uniform(-0.08, 0.08, 2)
        boxes.append(Box("climb", cx0 + jx, cy0 + jy, float(tier), side / 2, side / 2, h / 2, density))
    # Resolve z per stack: tier 0 sits on the floor, tier k sits on tier k-1's top face.
    by_stack: dict[tuple, list[Box]] = {}
    for b in boxes:
        by_stack.setdefault((round(b.cx, 1), round(b.cy, 1)), []).append(b)
    # Group by nearest stack center instead (jitter breaks exact (cx,cy) grouping).
    grouped = {i: [] for i in range(len(stack_centers))}
    for b in boxes:
        d = [(b.cx - c[0]) ** 2 + (b.cy - c[1]) ** 2 for c in stack_centers]
        grouped[int(np.argmin(d))].append(b)
    resolved = []
    for stack_i, blist in grouped.items():
        blist.sort(key=lambda b: b.cz)   # cz currently holds tier index
        top = PLINTH_TOP
        for b in blist:
            z = top + b.hz
            resolved.append(Box(b.zone, b.cx, b.cy, z, b.hx, b.hy, b.hz, b.density, b.yaw))
            top += 2 * b.hz
    return resolved


def sample_boxes(rng: np.random.Generator) -> list[Box]:
    """All 20 boxes for one round: fixed count and zone roles, sampled size/density/placement.
    `rng` should be seeded from the round seed (see env/sim.instance_spec's pattern) -- one draw
    per ROUND, not per instance, so every instance in a round faces the identical box field and
    only friction/wind (env/sim.py) vary between instances, exactly as upstream parkour does for
    its own geometry-is-fixed-per-round design."""
    return (_sample_scramble_boxes(rng) + _sample_push_boxes(rng) + _sample_climb_boxes(rng))


def build_course(rng: np.random.Generator):
    """Return (floor_segs, boxes, total_length). Mirrors parkour's build_course() shape so the
    referee/history/preview call sites need minimal changes."""
    floor = build_floor()
    boxes = sample_boxes(rng)
    return floor, boxes, ROOM_LENGTH


def floor_xml_fragment(segs: list[Seg]) -> str:
    out, i = [], 0
    for s in segs:
        for b in s.boxes:
            cx, cy, cz, hx, hy, hz, ck = b
            out.append(f'    <geom name="{GEOM_PREFIX}{i}" type="box" '
                       f'pos="{cx:.3f} {cy:.3f} {cz:.3f}" size="{hx:.3f} {hy:.3f} {hz:.3f}" '
                       f'condim="3" group="{WORLD_GROUP}" friction="0.9 .1 .1" rgba="{COLOR[ck]}"/>')
            i += 1
    return "\n".join(out)


def boxes_xml_fragment(boxes: list[Box]) -> str:
    """Boxes as FREE BODIES (joint type free) with density-derived mass/inertia, so pushing and
    climbing are genuine contact-solver outcomes, not scripted animations. Each gets its own
    friction band from its density (see _friction_band) so light push boxes are also slicker to
    stand on -- consistent with them being loose, low-friction crates rather than sticky ones."""
    out = []
    for i, b in enumerate(boxes):
        lo, hi = _friction_band(b.density)
        mu = lo + (hi - lo) * 0.5   # per-box median; sim.py may jitter like parkour's slabs
        out.append(
            f'  <body name="{BOX_PREFIX}{i}" pos="{b.cx:.3f} {b.cy:.3f} {b.cz:.3f}" '
            f'euler="0 0 {b.yaw:.4f}">\n'
            f'    <freejoint/>\n'
            f'    <geom name="{BOX_PREFIX}{i}_geom" type="box" '
            f'size="{b.hx:.3f} {b.hy:.3f} {b.hz:.3f}" density="{b.density:.2f}" '
            f'condim="3" group="{WORLD_GROUP}" friction="{mu:.4f} .1 .1" '
            f'rgba="{COLOR[b.zone]}"/>\n'
            f'  </body>'
        )
    return "\n".join(out)


if __name__ == "__main__":
    rng = np.random.default_rng([1, 0xB0B])
    floor, boxes, length = build_course(rng)
    print(f"room {length:.1f} m x {2 * TRACK_HALF_W:.1f} m, {len(boxes)} boxes\n")
    print(f"{'zone':10} {'x':>6} {'y':>6} {'z':>6} {'hx':>5} {'hy':>5} {'hz':>5} {'density':>8}")
    for b in boxes:
        print(f"{b.zone:10} {b.cx:6.2f} {b.cy:6.2f} {b.cz:6.2f} {b.hx:5.2f} {b.hy:5.2f} "
              f"{b.hz:5.2f} {b.density:8.1f}")
    by_zone: dict[str, int] = {}
    for b in boxes:
        by_zone[b.zone] = by_zone.get(b.zone, 0) + 1
    print(f"\nper-zone counts: {by_zone}")

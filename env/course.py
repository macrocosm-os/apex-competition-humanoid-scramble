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

ZONES, west to east along +x (see build_course) -- REVISED 2026-08-18 per Crux's vibe-check pass:
    0.0 - 2.0 m    start apron        clear; let the policy settle into gait before any obstacle
    2.0 - 12.0 m   MIXED field        scramble ("yellow"/orange) boxes are now scattered across
                                       the ENTIRE field -- not just their own sub-zone -- so small
                                       clutter shows up right through to the finish, not only at
                                       the start. Push (blue) boxes stay in their own corridor.
                                       Climb (red) boxes are restricted to the SECOND HALF of the
                                       room only (x >= 6.0 m) -- no red until the halfway point,
                                       then it's the dominant obstacle through to the dash finish.
    11.0 - 12.0 m  dash finish        clear straight sprint to the line

    Zone ROLE (which density/size band a box is sampled from, and therefore what verb it forces)
    is still fixed and still assigned by the same fixed counts (9 scramble / 5 push / 6 climb --
    see N_SCRAMBLE/N_PUSH/N_CLIMB below, unchanged). What changed is PLACEMENT: a box's x-position
    is no longer confined to one contiguous sub-zone per role. See _sample_scramble_boxes,
    _sample_push_boxes, _sample_climb_boxes for the interleaved placement bands.

    This mirrors the linear parkour course's own principle (progress-based scoring needs a
    continuous difficulty gradient, not discrete tiers) while giving the room-crossing brief its
    four verbs: dash (start apron + finish), scramble (dense small clutter, now everywhere),
    push (light boxes in-path), climb (heavy stacked boxes, now back-loaded to the second half so
    difficulty still ramps up toward the finish rather than flattening once zones interleave).

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
TRACK_HALF_W = 3.0            # 6 m room width -- UNCHANGED again (2026-08-18, second length-only
                               # doubling ask): only ROOM_LENGTH has ever been asked to double;
                               # width has never been touched, so the room keeps getting longer
                               # and relatively narrower (was 4:1, now 8:1). Still correct per the
                               # literal instruction each time -- flagging again here because two
                               # consecutive length-doublings without a width change is worth a
                               # human sanity check on whether that's still intended.
ROOM_LENGTH = 48.0             # 2026-08-18 (Crux, second doubling): 24.0 -> 48.0 m.

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
# NOTE (2026-08-18 revision): these lengths now describe each role's SAMPLING BAND, not a
# dedicated sub-zone it's confined to -- see the module docstring's ZONES section. Scramble's
# band is the full field (interleaved everywhere); push keeps its own corridor; climb's band is
# pushed to the second half of the room only.
# 2026-08-18 (Crux): room length doubled 12.0 -> 24.0 m. Every zone length below scaled x2 in
# lockstep so each phase (apron/field/push-corridor/dash) still occupies the same FRACTION of the
# room it did before, rather than e.g. leaving the apron at a fixed 2.0 m and only stretching the
# field -- that would have changed the course's pacing (how much of the run is warm-up vs.
# obstacle vs. sprint-finish), which was not asked for. Pure x2 scale-up, same proportions.
# 2026-08-18 (Crux, second doubling): APRON and FIELD doubled again in lockstep with ROOM_LENGTH,
# same reasoning as the first doubling (keep each phase's FRACTION of the room constant). DASH is
# the exception this time: Crux explicitly asked for "a short sprint to the finish line" as a new,
# distinct requirement -- not "double the dash length too" -- so DASH_LEN is set to a fixed short
# value (3.0 m) rather than scaled with the room, long enough to read as a real final sprint
# phase (a few strides at G1 scale) but deliberately NOT proportioned to the 48 m room the way the
# apron/field split still is.
APRON_LEN = 8.0                 # was 4.0
FIELD_LEN = 37.0                # was 18.0, adjusted so APRON+FIELD+DASH == ROOM_LENGTH exactly
                                  # with the new fixed-length DASH below (8.0 + 37.0 + 3.0 = 48.0)
DASH_LEN = 3.0                   # was 2.0 -- now a fixed "short sprint" length, not room-scaled
assert APRON_LEN + FIELD_LEN + DASH_LEN == ROOM_LENGTH

SCRAMBLE_X0, SCRAMBLE_LEN = APRON_LEN, FIELD_LEN                    # 4.0 - 22.0 m: everywhere
# Push (blue) corridor widened to the WHOLE field (2026-08-18, Crux: "spread the blue boxes out
# further") -- previously a fixed 6 m sub-span (10.0-16.0 m) sized as a fraction of the old 12 m
# room; now spans the same 4.0-22.0 m band as scramble so 15 (soon more, see N_PUSH below) blue
# boxes aren't all crammed into one 6 m stretch that's also where climb's stacks start.
PUSH_X0, PUSH_LEN = APRON_LEN, FIELD_LEN                            # 4.0 - 22.0 m: whole field,
                                                                      # same band as scramble
# Second half of the room is x >= ROOM_LENGTH / 2 = 12.0 m (scales automatically with
# ROOM_LENGTH -- this was already a formula, not a hardcoded 6.0, so the "only red in the second
# half" rule from Crux's earlier request holds unchanged at the new room length).
CLIMB_X0 = ROOM_LENGTH / 2
CLIMB_LEN = (APRON_LEN + FIELD_LEN) - CLIMB_X0

ZONE_X0 = {"scramble": SCRAMBLE_X0, "push": PUSH_X0, "climb": CLIMB_X0}
ZONE_LEN = {"scramble": SCRAMBLE_LEN, "push": PUSH_LEN, "climb": CLIMB_LEN}

# Fixed box count per zone -- the brief's "fixed number of boxes", split by role. Total 20,
# derived from the ~20% floor-packing target above. NOT sampled: this is deterministic so the
# course's difficulty shape is stable round to round; only each box's size/density/exact
# placement jitter is drawn from the round seed.
# 2026-08-18 (Crux): box count tripled 20 -> 60, exactly x3 per zone role so the 9:5:6 role mix
# (scramble:push:climb) -- and therefore the relative amount of each verb -- is unchanged, only
# the absolute count is. This keeps ~20% floor-packing density roughly intact too: floor area
# also doubled (length x2, width unchanged) to 6 x 24 = 144 sq m, and box footprint area tripled,
# so packing fraction goes from ~20% to ~30% of floor area -- deliberately denser, consistent
# with Crux's separate "increase density" ask earlier in this thread, not a bug to correct.
# 2026-08-18 (Crux): "add more of the blue and yellow boxes" -- scramble (yellow) and push (blue)
# counts bumped up from the x3 rescale (27/15) while climb (red) stays at 18 (not asked to grow).
# Chosen to roughly double each of yellow/blue's PRIOR count rather than an arbitrary bump, so the
# increase reads as deliberate escalation, not noise: 27->50 yellow, 15->30 blue.
# 2026-08-18 (Crux, second doubling): "proportionally increase the number of objects" to match the
# room-length doubling (24 -> 48 m, x2 floor area since width is unchanged) -- scaling all three
# counts x2 keeps the same 50:30:18 role ratio (so the same relative amount of scramble/push/climb
# difficulty) and keeps floor-packing fraction roughly where it was after the last count bump,
# rather than thinning out over the now-doubled floor area.
N_SCRAMBLE, N_PUSH, N_CLIMB = 100, 60, 36   # was 50, 30, 18 -- exactly x2 each
N_BOXES = N_SCRAMBLE + N_PUSH + N_CLIMB   # 196

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

# Climb (red) boxes: median/min/max scaled x1.5 over the original band (0.70/0.45/1.00 ->
# 1.05/0.675/1.50) per Crux's 2026-08-18 vibe-check note. At this size a single climb box's top
# sits above what a G1 can reach with a one-leg step-up (upstream's proven ceiling is 0.55 m
# single-tier), so getting on top now genuinely requires EITHER a real climb (hands/torso-assisted
# mount, or using an adjacent smaller scramble/push box as a step) rather than the old
# single-step-onto-tier-1 path. Height keeps its own +50% too so a lone climb box is a real
# obstacle on its own, not just a wide plinth.
# 2026-08-18 (Crux, this change): "allow the red blocks to be up to 30% bigger" -- read as widening
# the UPPER bound of the existing band (not shifting median/lo, which would make EVERY red box
# bigger rather than allowing the biggest ones to get bigger). max side 1.50 -> 1.95 (x1.3), max
# height 0.69 -> 0.897 (x1.3). Median/sigma/min untouched, so most climb boxes look similar to
# before and only the long tail of the lognormal draw reaches the new, taller ceiling.
CLIMB_SIDE = (1.05, 0.18, 0.675, 1.95)          # max was 1.50, now x1.3 = 1.95
CLIMB_HEIGHT = (0.51, 0.15, 0.39, 0.897)        # max was 0.69, now x1.3 = 0.897; per tier -- see
                                                  # CLIMB_MAX_TIERS/CLIMB_MIN_TIERS for stacking

# Density x1.5 across the board, per Crux's 2026-08-18 note ("increase density of all blocks by
# 50%"). Scaling the whole (lo, hi) band rather than re-deriving from anchors keeps every zone's
# relative ordering (scramble < push < climb) and each zone's real-world material analogy intact
# -- push boxes are still the lightest, climb boxes still the densest -- just uniformly heavier.
_DENSITY_SCALE = 1.5

# Density bands, kg/m^3. Calibrated against what a ~32 kg legs-only G1 can plausibly move with a
# body-check vs. what it needs as stable footing (docs/design.md, "push corridor sizing" and
# "climb stack sizing"). Real-world anchors: dry foam/cardboard ~15-60, packed textiles/light
# plastics ~150-350, water ~1000, wet sand/dense-packed goods ~1400-1900, similar to a
# loaded packing crate.
DENSITY_SCRAMBLE = (40.0 * _DENSITY_SCALE, 260.0 * _DENSITY_SCALE)   # light clutter, still easy
                                                                       # to knock relative to push/climb
DENSITY_PUSH = (25.0 * _DENSITY_SCALE, 90.0 * _DENSITY_SCALE)         # still deliberately the
                                                                       # LOWEST band -- shovable by
                                                                       # contact force alone, just
                                                                       # heavier in absolute terms
DENSITY_CLIMB = (350.0 * _DENSITY_SCALE, 1400.0 * _DENSITY_SCALE)     # still deliberately HIGHEST:
                                                                       # stable footing, doesn't slide/tip

# Sliding friction scales with density (heavier, denser material grips better in this course's
# fiction — think rubberised crate vs. slick lightweight tote), same spirit as parkour's
# friction-band-by-surface-kind. Returns (lo, hi) mu band for a given density.
def _friction_band(density: float) -> tuple[float, float]:
    lo = float(np.interp(density, [20.0, 1400.0], [0.15, 0.55]))
    hi = float(np.interp(density, [20.0, 1400.0], [0.35, 0.95]))
    return lo, hi


# ---------------------------------------------------------------------------------------------
# OVERLAP PREVENTION (2026-08-18, Crux: "make sure no objects overlap")
#
# Every _sample_*_boxes function up to now placed boxes independently -- one per x-slice/lane-
# slot with only small position jitter -- with no check against OTHER already-placed boxes, in
# its own zone or any other. At the box counts before this change (9/5/6, then 27/15/18, then
# 50/30/18) that was rarely visible because the placement grids were sparse relative to box size;
# at 100/60/36 (this change) with boxes also now bigger, XY footprint overlap becomes actually
# likely, not just theoretically possible, so this is being fixed as a real placement constraint
# rather than left as an assumption.
#
# Approach: maintain a single flat list of already-placed (cx, cy, hx, hy, yaw=0-ish) footprints
# across ALL zones (scramble+push+climb share one registry, sampled in that order), and reject/
# resample any candidate whose axis-aligned bounding box -- inflated by a small clearance margin,
# not just touching -- overlaps an existing one. Climb tiers are the one deliberate exception:
# tier k of a stack is SUPPOSED to sit on top of tier k-1 (same XY footprint, different Z), so the
# registry only tracks tier-0 (base) footprints per stack, not every tier -- checking every tier
# against itself would make stacking impossible, which is the opposite of a bug.
#
# This is rejection sampling, not a packing solver: at high fill fractions it can need many
# retries per box. MAX_PLACEMENT_ATTEMPTS bounds that; if the field is too dense to place
# everything (should not happen at this course's sizes/counts, but a genuine capacity limit is a
# real possibility other future scale-ups could hit), a box is placed at its last-tried position
# rather than crashing course generation -- an assertion would turn a rare tight-packing case into
# a hard failure for an otherwise-fine round; logging would be silent and easy to miss. Overlap in
# that rare fallback case is a soft degradation (two boxes touching/interpenetrating slightly),
# not a broken course, and is intentionally the worst case rather than the common one.
OVERLAP_MARGIN = 0.06            # metres of clearance required between any two box footprints
MAX_PLACEMENT_ATTEMPTS = 60       # retries before falling back to the last candidate position


def _overlaps(cx: float, cy: float, hx: float, hy: float, yaw: float,
              placed: list[tuple[float, float, float, float, float]]) -> bool:
    """AABB overlap test in the box's own (slightly yaw-rotated) footprint vs. every placed
    footprint, both inflated by OVERLAP_MARGIN. Yaw jitter on these boxes is small (+-0.4 rad
    max) so an axis-aligned check with a bit of extra margin is a good enough approximation of
    the true rotated-rectangle overlap -- exact SAT collision isn't worth the complexity here."""
    ax0, ax1 = cx - hx - OVERLAP_MARGIN, cx + hx + OVERLAP_MARGIN
    ay0, ay1 = cy - hy - OVERLAP_MARGIN, cy + hy + OVERLAP_MARGIN
    for (pcx, pcy, phx, phy, _pyaw) in placed:
        bx0, bx1 = pcx - phx, pcx + phx
        by0, by1 = pcy - phy, pcy + phy
        if ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0:
            return True
    return False


def _place_no_overlap(rng: np.random.Generator, hx: float, hy: float,
                       placed: list[tuple[float, float, float, float, float]],
                       sample_xy) -> tuple[float, float]:
    """Repeatedly draw (cx, cy) from `sample_xy()` (a zero-arg callable capturing whatever
    per-box x/y sampling distribution the caller wants) until one doesn't overlap any footprint
    already in `placed`, or MAX_PLACEMENT_ATTEMPTS is exhausted. Appends the accepted footprint
    to `placed` (mutates in place) so the NEXT call sees this box too -- callers must place boxes
    one at a time through this function for the registry to actually prevent overlaps."""
    cx = cy = 0.0
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        cx, cy = sample_xy()
        if not _overlaps(cx, cy, hx, hy, 0.0, placed):
            break
    placed.append((cx, cy, hx, hy, 0.0))
    return cx, cy


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
    the policy can rely on.

    BUG FIX (2026-08-18): this used to compute the field slab's length as
    SCRAMBLE_LEN + PUSH_LEN + CLIMB_LEN, which was correct back when each of those constants was
    its own disjoint sub-zone length that summed to the field. After the 2026-08-18 interleaving
    change, SCRAMBLE_LEN became the length of the WHOLE field (boxes now placed everywhere, not a
    sub-zone) while PUSH_LEN/CLIMB_LEN stayed their own (now-overlapping) sampling bands -- so the
    old sum (9.0 + 3.0 + 5.0 = 17.0 m) put the field slab, and everything after it (the dash strip,
    the finish line at ROOM_LENGTH), 5 m past the actual 12 m room. This is why the finish line
    didn't read as being at the end of the visible course (Crux, 2026-08-18) -- the floor really
    was 5 m longer than the scored room. Use FIELD_LEN directly: it is the one authoritative
    field-length constant now (see the zone-extents block above), not a sum of the per-role bands.
    """
    segs = []
    segs.append(Seg("apron", APRON_LEN, [_slab(0.0, APRON_LEN, PLINTH_TOP, "apron")]))
    segs.append(Seg("field", FIELD_LEN,
                    [_slab(APRON_LEN, FIELD_LEN, PLINTH_TOP, "apron")]))
    # Dash slab starts right after apron+field (== APRON_LEN + FIELD_LEN, not a re-sum of the
    # per-role bands -- SCRAMBLE_LEN/PUSH_LEN/CLIMB_LEN overlap each other by design since the
    # 2026-08-18 interleaving change, so summing them here would double/triple-count and push the
    # dash slab (and therefore the finish line) past ROOM_LENGTH again, the exact bug already
    # fixed once above; use the same APRON_LEN + FIELD_LEN this function already computed with.
    segs.append(Seg("dash", DASH_LEN,
                    [_slab(APRON_LEN + FIELD_LEN, DASH_LEN, PLINTH_TOP, "dash")]))
    return segs


def _lognormal_clip(rng: np.random.Generator, median: float, sigma: float, lo: float, hi: float) -> float:
    """Lognormal draw with a given median (not mean), clipped to [lo, hi]. Lognormal rather than
    uniform/normal because real clutter/crate size distributions are right-skewed -- lots of
    smallish boxes, a long thin tail of big ones -- and it can never draw a negative size."""
    val = float(rng.lognormal(mean=math.log(median), sigma=sigma))
    return float(np.clip(val, lo, hi))


def _sample_scramble_boxes(rng: np.random.Generator, placed: list) -> list[Box]:
    """Small/medium clutter ("yellow"), spread across the WHOLE field (2.0-11.0 m), not confined
    to a leading sub-zone -- per Crux's 2026-08-18 note, yellow boxes now show up throughout the
    entire course rather than only at the start, so the push corridor and the climb-stack second
    half both still have loose small clutter woven in around their own zone's boxes, not just
    clean floor between piles.

    GUARANTEED even coverage (2026-08-18 fix): with only 9 boxes free to land anywhere across a
    9 m band, a plain random/shuffled slot draw can (and in testing did) bunch most of them into
    one 2-3 m stretch and leave 2+ metres bare at either end -- exactly the failure mode Crux's
    vibe-check renders caught ("yellow only in the middle, not really throughout"). Fixed by
    partitioning the field into N_SCRAMBLE equal-width x-SLICES first (one box guaranteed per
    slice, so no >1-slice-wide gap is possible) and only randomising placement WITHIN each slice
    (x jitter + full-width y draw) -- coverage is structural, not a hope from shuffling."""
    boxes = []
    x0 = ZONE_X0["scramble"]
    slice_len = SCRAMBLE_LEN / N_SCRAMBLE
    ys = np.linspace(-TRACK_HALF_W + 0.5, TRACK_HALF_W - 0.5, 5)
    for i in range(N_SCRAMBLE):   # noqa: keep loop var name stable across the diff
        slice_x0 = x0 + i * slice_len
        side = _lognormal_clip(rng, *SCRAMBLE_SIDE)
        h = _lognormal_clip(rng, *SCRAMBLE_HEIGHT)
        density = float(rng.uniform(*DENSITY_SCRAMBLE))
        yaw = float(rng.uniform(-0.4, 0.4))

        def sample_xy(slice_x0=slice_x0, slice_len=slice_len):
            sx = slice_x0 + slice_len / 2 + float(rng.uniform(-slice_len * 0.3, slice_len * 0.3))
            sy = float(rng.choice(ys)) + float(rng.uniform(-0.3, 0.3))
            sy = float(np.clip(sy, -TRACK_HALF_W + 0.3, TRACK_HALF_W - 0.3))
            return sx, sy

        # No-overlap placement (2026-08-18, Crux): retries within this box's own slice/lane
        # distribution until it clears every box placed so far (any zone, since `placed` is
        # shared -- see sample_boxes). Slice-based x already guarantees coverage; this only adds
        # the overlap rejection on top, so the earlier "guaranteed even spread" fix still holds.
        sx, sy = _place_no_overlap(rng, side / 2, side / 2, placed, sample_xy)
        boxes.append(Box("scramble", sx, sy, PLINTH_TOP + h / 2,
                        side / 2, side / 2, h / 2, density, yaw))
    return boxes


def _sample_push_boxes(rng: np.random.Generator, placed: list) -> list[Box]:
    """Low-density boxes placed IN the lane, one per lane-width slice, so displacing one is the
    intended solution: the only alternative is squeezing past in a track that is deliberately
    narrower than a G1's shoulder-width clearance once a box sits at lane centre.

    Spread tightened to the corridor's full length (2026-08-18): previously bunched inside a
    2 m sub-span of its own 3 m corridor (0.5 m margins each side left only 2 m of the 3 m
    corridor actually used), which combined with climb's overlapping x-range read as one dense
    mid-course clump in the vibe-check renders rather than push's own distinct phase. Margins
    trimmed to 0.3 m so all 5 boxes use close to the full PUSH_LEN corridor."""
    # LATERAL (y) spread fix (2026-08-18, Crux: "why are blue boxes always in the middle... look
    # at the y axis"): the old y formula -- uniform(-0.6, 0.6) plus an optional +-0.5 nudge --
    # tops out at |y| ~= 1.1 no matter how many boxes there are, because it was sized for the
    # ORIGINAL 5-box push zone at the old 12 m room and never revisited when N_PUSH grew (5 -> 15
    # -> 30). With TRACK_HALF_W = 3.0 m (track is -3.0 to 3.0), that left over half the width on
    # each side permanently empty of blue boxes -- a real bug, not a rendering illusion (confirmed
    # against raw sampled y-values, not just the image). Replaced with a full-width lane draw
    # (same [-TRACK_HALF_W+0.4, TRACK_HALF_W-0.4] range scramble already uses) so blue boxes can
    # land anywhere across the room's actual width, not just a ~2 m centre strip.
    boxes = []
    x0 = ZONE_X0["push"]
    xs = np.linspace(x0 + 0.3, x0 + PUSH_LEN - 0.3, N_PUSH)
    for sx0 in xs:
        side = _lognormal_clip(rng, *PUSH_SIDE)
        h = _lognormal_clip(rng, *PUSH_HEIGHT)
        density = float(rng.uniform(*DENSITY_PUSH))

        def sample_xy(sx0=sx0):
            # Small x jitter around this box's own lane slot, on top of the full-width y draw --
            # gives the no-overlap retry loop (below) some room to move in BOTH axes when the
            # first draw collides, rather than only ever retrying y.
            sx = sx0 + float(rng.uniform(-0.4, 0.4))
            sy = float(rng.uniform(-TRACK_HALF_W + 0.4, TRACK_HALF_W - 0.4))
            return sx, sy

        sx, sy = _place_no_overlap(rng, side / 2, side / 2, placed, sample_xy)
        boxes.append(Box("push", sx, sy, PLINTH_TOP + h / 2, side / 2, side / 2, h / 2, density))
    return boxes


CLIMB_MAX_TIERS = 2   # was 2-3; capped at 2 (2026-08-18, Crux) -- see docstring below. Still the
                       # CEILING on tiers per stack; see CLIMB_MIN_TIERS below for the 2026-08-18
                       # "not always stacked in twos" change that makes the tier count PER STACK
                       # variable between 1 and this ceiling, rather than always exactly this.
CLIMB_MIN_TIERS = 1    # 2026-08-18 (Crux: "not always stacked in twos"): a stack can now resolve
                       # to a single lone box (no second tier at all) as often as to a genuine
                       # 2-tier stack -- see the per-stack tier-count draw in _sample_climb_boxes.
                       # This also directly enables the brief's OTHER new climb requirement:
                       # "require climbing OR stacking a smaller box to get on top" already held
                       # for 2-tier stacks (mount tier 1, stand, mount tier 2), but a lone
                       # oversized (now up to 30% bigger, see CLIMB_SIDE_MAX_BONUS below) box with
                       # no second tier is the case where a nearby smaller scramble/push box is
                       # the ONLY way up without a true climb -- variety Crux asked for by name.


def _sample_climb_boxes(rng: np.random.Generator, placed: list) -> list[Box]:
    """Heavy, broad, squat ("red") boxes, confined to the SECOND HALF of the room only
    (x >= 6.0 m, i.e. CLIMB_X0) per Crux's 2026-08-18 note -- no red until the halfway point, then
    it is the dominant obstacle through to the dash finish.

    Stack centres now spread across the FULL second-half band (2026-08-18): previously packed
    into a ~1.5 m window near the front of the climb band, which visually fused with the
    overlapping push corridor into a single mid-course clump rather than reading as its own
    zone continuing toward the finish. Spread across CLIMB_LEN so the last stack sits close to
    the dash finish, not bunched right after push.

    STACK CAP (2026-08-18 revision): capped at CLIMB_MAX_TIERS=2, not the old 2-3. The old 3-tier
    piles were flagged in the vibe-check render as "upper piled red boxes" -- a third box balanced
    on top read as precarious/cluttered rather than as one clean obstacle, and did not add a
    distinct new skill over a 2-tier mount. With CLIMB_SIDE/CLIMB_HEIGHT now themselves +50
    (module-level change above), a SINGLE climb box's top already sits at ~0.65-1.0 m -- above
    upstream's proven single-leg step-up ceiling (0.55 m) on its own -- so climbing one enlarged
    box, or two stacked, is already "too tall to step onto directly" without needing a third tier.
    A lone tall climb box also now reads as legitimately mountable via an adjacent smaller
    scramble/push box used as a step (the brief's "stacking a smaller box to get on top" path),
    which the interleaved placement (scramble boxes now scattered through this half too) makes
    available where it wasn't when scramble was confined to the front of the room.

    Stack-packing: each stack's footprint shrinks going up (base tier widest) so the pile is
    self-stable under gravity without scripted joints -- the same reason a real box stack is
    pyramided. Two stacks side by side leave a narrow gap between them as an alternate,
    harder-to-balance route for a policy that would rather thread the middle than climb either."""
    # Stack centres spread across the FULL CLIMB_LEN band (2026-08-18 rescale): with 18 climb
    # boxes now (was 6) and CLIMB_MAX_TIERS=2, that's 9 stack footprints needed, not 3 -- laying
    # out 9 x0-offsets spanning CLIMB_LEN (rather than 3 fixed offsets sized for the old 3 m band)
    # is what keeps climb boxes spread across the whole second half at the new 24 m room length,
    # instead of 9 stacks all landing on top of the same 3 old offsets near CLIMB_X0.
    # LATERAL (y) spread fix (2026-08-18, Crux: "red boxes also confined to the center... look at
    # the y axis"): stack_ys used to cycle through only 3 fixed offsets (-1.4, 1.4, 0.0), sized
    # for the ORIGINAL 3-stack layout at the old 12 m room and never revisited when the stack
    # count grew with N_CLIMB. With TRACK_HALF_W = 3.0 m that left the outer ~1.5 m on each side
    # permanently empty of red boxes, same bug class as the push-zone fix just above. Now draws
    # n_footprints evenly spaced y offsets across most of the track width (leaving a small margin
    # so stacks don't clip the side walls), so climb stacks spread laterally too, not just along x.
    # Minimum stack-to-stack separation (2026-08-18, Crux: "make sure no objects overlap"), FIXED
    # (second pass): the first version of this fix reserved CLIMB_SIDE_MAX (1.95 m, the absolute
    # lognormal ceiling) as every stack's footprint, worst-case-for-all. At N_CLIMB=36 with
    # CLIMB_MIN_TIERS=1 (most stacks are now single boxes, not always-2 -- see the "not always
    # stacked in twos" change), that worst-case reservation needed ~154 sq m of packing area in a
    # climb band that only HAS ~92 sq m (CLIMB_LEN x usable width) -- ran the numbers after the
    # first fix still showed overlaps and this is why: rejection sampling was exhausting its
    # retries and falling back to overlapping placements because the reservation was, structurally,
    # too big to ever satisfy at this box count -- not a rare edge case, the COMMON case at N=36.
    # Fixed by reserving each footprint at the size it will ACTUALLY typically need (median side,
    # not max) with a smaller safety pad, and separately relying on _overlaps' exact (not
    # reservation-based) check at the point each box is finally emitted (see the real per-box
    # emission loop below, which re-validates against `placed` using each box's REAL sampled
    # size, not the reservation) to catch the rarer big-draw case without requiring every stack to
    # pre-reserve room for the largest possible neighbour.
    CLIMB_SIDE_MEDIAN = CLIMB_SIDE[0]   # 1.05 m -- realistic per-stack footprint budget
    STACK_RESERVE_HALF = CLIMB_SIDE_MEDIAN / 2 + OVERLAP_MARGIN

    boxes = []
    x0 = ZONE_X0["climb"]
    n_footprints = -(-N_CLIMB // CLIMB_MIN_TIERS)   # WORST case (all stacks end up 1-tier): use
                                                      # CLIMB_MIN_TIERS so there are always enough
                                                      # footprint SLOTS reserved; actual per-stack
                                                      # tier counts are drawn below and can vary.
    stack_centers: list[tuple[float, float]] = []

    def stack_xy():
        return (float(rng.uniform(x0 + 0.8, x0 + CLIMB_LEN - 0.8)),
                float(rng.uniform(-TRACK_HALF_W + 0.8, TRACK_HALF_W - 0.8)))

    for _ in range(n_footprints):
        cx, cy = _place_no_overlap(rng, STACK_RESERVE_HALF, STACK_RESERVE_HALF, placed, stack_xy)
        stack_centers.append((cx, cy))

    # Per-stack tier count (2026-08-18, Crux: "not always stacked in twos"): each footprint slot
    # independently draws how many tiers it will actually get (CLIMB_MIN_TIERS..CLIMB_MAX_TIERS,
    # inclusive) BEFORE boxes are assigned to it, rather than every stack silently filling up to
    # CLIMB_MAX_TIERS whenever enough boxes are available (the old behaviour, which is why every
    # stack ended up 2-tier in practice even though the cap was nominally a ceiling, not a target).
    # Stops assigning to a slot once it hits its own drawn cap; leftover boxes spill into starting
    # NEW slots beyond n_footprints as needed, so no climb box goes unplaced.
    slot_caps = [int(rng.integers(CLIMB_MIN_TIERS, CLIMB_MAX_TIERS + 1)) for _ in stack_centers]
    per_stack = [0] * len(stack_centers)
    order = list(range(N_CLIMB))
    rng.shuffle(order)
    for i in order:
        # Find the next slot with room under ITS OWN drawn cap; extend with a fresh slot (new x/y,
        # fresh random cap) if every existing slot is already full -- keeps the "not always two"
        # variability even when N_CLIMB doesn't divide evenly across the reserved footprints.
        stack_i = next((k for k in range(len(stack_centers)) if per_stack[k] < slot_caps[k]), None)
        if stack_i is None:
            stack_i = len(stack_centers)
            extra_x = float(rng.uniform(x0 + 0.8, x0 + CLIMB_LEN - 0.8))
            extra_y = float(rng.uniform(-TRACK_HALF_W + 0.8, TRACK_HALF_W - 0.8))
            stack_centers.append((extra_x, extra_y))
            slot_caps.append(int(rng.integers(CLIMB_MIN_TIERS, CLIMB_MAX_TIERS + 1)))
            per_stack.append(0)
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

    # No-overlap check ACROSS stacks (2026-08-18, Crux: "make sure no objects overlap"), FIXED
    # (second pass): the first version only registered the RESERVATION placeholder size
    # (STACK_RESERVE_HALF, sized off the median) into `placed`, never the base tier's REAL
    # sampled half-extent -- so a stack whose base tier happened to draw close to CLIMB_SIDE's
    # max (1.95 m) had a footprint bigger than what was reserved for it, and could genuinely
    # overlap a push/scramble box placed nearby that only checked against the (too-small)
    # reservation. Fixed by re-registering with the ACTUAL base half-extent once it's known, and
    # nudging the stack's centre (small local search) if the real footprint doesn't clear its
    # neighbours at the reserved position -- rare in practice since most stacks draw near the
    # median, but this is exactly the tail case that needs a real fix, not a bigger reservation
    # (which is what caused the earlier capacity-exhaustion failure at N_CLIMB=36).
    for stack_i, (cx0, cy0) in enumerate(stack_centers):
        base = next((b for b in boxes if b.cz == 0.0 and abs(b.cx - cx0) < 0.15
                     and abs(b.cy - cy0) < 0.15), None)
        if base is None:
            continue
        # `placed` still contains this stack's own RESERVATION entry (added by _place_no_overlap
        # above) -- exclude it before re-checking with the real size, or the real footprint would
        # always "overlap" its own placeholder.
        others = [p for p in placed if not (abs(p[0] - cx0) < 1e-6 and abs(p[1] - cy0) < 1e-6)]
        fx, fy = cx0, cy0
        if _overlaps(fx, fy, base.hx, base.hy, 0.0, others):
            # Real footprint collides at the reserved centre: try small local nudges before
            # giving up (this is the rare tail case noted above, not the common path).
            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                nx = cx0 + float(rng.uniform(-0.5, 0.5))
                ny = cy0 + float(rng.uniform(-0.5, 0.5))
                if not _overlaps(nx, ny, base.hx, base.hy, 0.0, others):
                    fx, fy = nx, ny
                    break
            for i, b in enumerate(boxes):
                if b.cz == 0.0 and abs(b.cx - cx0) < 0.15 and abs(b.cy - cy0) < 0.15:
                    boxes[i] = Box(b.zone, fx, fy, b.cz, b.hx, b.hy, b.hz, b.density, b.yaw)
            stack_centers[stack_i] = (fx, fy)
        placed.append((fx, fy, base.hx, base.hy, 0.0))

    # Resolve z per stack: tier 0 sits on the floor, tier k sits on tier k-1's top face.
    # Group by nearest stack center (jitter breaks exact (cx,cy) grouping).
    grouped = {i: [] for i in range(len(stack_centers))}
    for b in boxes:
        d = [(b.cx - c[0]) ** 2 + (b.cy - c[1]) ** 2 for c in stack_centers]
        grouped[int(np.argmin(d))].append(b)
    resolved = []
    for stack_i, blist in grouped.items():
        if not blist:
            continue
        # Sort by ACTUAL sampled footprint (largest first), not by the random tier index each
        # box was assigned, same reasoning as before the tier cap: lognormal sampling noise can
        # still hand a later tier a bigger raw side than an earlier one. Sorting by measured
        # footprint enforces "widest at the bottom" independent of assignment order.
        blist.sort(key=lambda b: -(b.hx * b.hy))
        top = PLINTH_TOP
        for b in blist:
            z = top + b.hz
            resolved.append(Box(b.zone, b.cx, b.cy, z, b.hx, b.hy, b.hz, b.density, b.yaw))
            top += 2 * b.hz
    return resolved


def sample_boxes(rng: np.random.Generator) -> list[Box]:
    """All boxes for one round: fixed count and zone roles, sampled size/density/placement.
    `rng` should be seeded from the round seed (see env/sim.instance_spec's pattern) -- one draw
    per ROUND, not per instance, so every instance in a round faces the identical box field and
    only friction/wind (env/sim.py) vary between instances, exactly as upstream parkour does for
    its own geometry-is-fixed-per-round design.

    OVERLAP PREVENTION (2026-08-18, Crux: "make sure no objects overlap"): `placed` is a single
    footprint registry SHARED across all three zone samplers, passed in call order (climb first,
    then push, then scramble) so later-sampled zones' boxes are rejected/resampled against
    EARLIER zones' footprints too, not just their own zone's. Climb goes first deliberately: its
    stack positions are the most constrained (fixed second-half x band, discrete footprint slots)
    and least tolerant of being pushed around by a retry loop, so it gets first claim on floor
    space; push and scramble (more numerous, smaller, more flexible placement bands) fit around
    whatever climb has already claimed."""
    placed: list[tuple[float, float, float, float, float]] = []
    climb = _sample_climb_boxes(rng, placed)
    push = _sample_push_boxes(rng, placed)
    scramble = _sample_scramble_boxes(rng, placed)
    return climb + push + scramble


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

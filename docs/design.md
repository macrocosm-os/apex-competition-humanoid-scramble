# Design decisions

Fork of Humanoid Parkour's `docs/design.md`. What the code does is in the code; this records what
is not visible there — what was measured, what was rejected, and what is still open. Decisions
inherited unchanged from upstream (robot choice, on-ramp calibration methodology, wind model,
recurrence requirement, "compile once" optimisation pattern, rejected checkpoint scoring / energy
budget / hands-on-obstacles) are not re-litigated here; only what this fork changed or added.

## Why fork Humanoid Parkour rather than start clean

Same robot (Unitree G1, legs-only), same `gym_v1` obs[104]/state[256]/action[12] interface, same
referee/player split, same progress-based scoring shape. What changes is the geometry: instead of
a fixed sequence of named maneuvers, the robot crosses a room scattered with a per-round-sampled
box field. Reusing the interface means a miner's understanding of the perception channels (height
scan, overhead clearance, gait clock, heading) transfers directly, and the resource/timeout
budgeting upstream already measured (referee ~560 MiB, ~1 CPU average, 15 MB submission cap) is a
sound starting prior rather than a guess — though it MUST be re-measured before onboarding (see
"What changed the cost model" below), because box contact physics is not free-geometry physics.

## Room geometry: why 12 m x 6 m (aspect 2:1)

The brief specified aspect 2:1 and asked for the ratio of room size to object count to be reasoned
through, not guessed. Worked through explicitly rather than picked by feel:

- **Width sets lateral routing choice.** Upstream's corridor is 2.4 m wide (TRACK_HALF_W 1.2 m) —
  correct for a linear maneuver sequence with no routing decision, wrong for a room where "go
  around, or through" is supposed to be a live choice. 6 m (TRACK_HALF_W 3.0 m) gives three ~2 m
  lanes: enough for a genuine sidestep detour around one pile without being so wide that a policy
  can always find clear floor and never has to interact with a box. This is the width that makes
  clustering (see zone design below) *force* something rather than merely offer it as an option.
- **Length = 2x width**, per the brief's aspect ratio, giving 12 m. Split into a start apron, three
  obstacle zones, and a dash finish (below) — long enough for each zone to read as a distinct
  phase of the crossing, short enough to stay well inside the wall-clock budget upstream parkour
  was already sized against.

## Box count: why 20, and why fixed

**The packing-fraction argument.** Floor area is 6 x 12 = 72 sq m. A course that is all dash lanes
never forces scrambling; a course that is wall-to-wall boxes is a wall, not an obstacle. Target
packing fraction ~20% of floor area covered by box footprints (non-uniformly distributed —
clustered per zone, not gridded, so real open floor exists in between and at both ends). Average
sampled box footprint area works out to ~0.72 sq m ((0.85 m avg side)^2). At 20% packing:
0.20 x 72 / 0.72 ~= 20 boxes. Split deterministically 9 scramble / 5 push / 6 climb by role (see
zone design), which is the "fixed number of boxes" in the brief — what's sampled per round is
each box's size, density, and placement jitter, not the count or the zone roster.

**Why fixed, not itself sampled per round — the anti-Goodhart argument.** If box count were drawn
per round, `raw_score` variance would blend two different things: how skilled a policy is, and how
many boxes it happened to get. A round with fewer boxes would inflate every submission's score
that round regardless of skill — the exact round-to-round confound upstream's own docs worry about
for friction and wind (see "Randomised conditions on a fixed course" below, inherited). Fixing the
count keeps the difficulty *envelope* stable and channels all round-to-round variance into what
the design already budgets for: per-box size/density/placement and per-instance wind. This mirrors
upstream's own split exactly — geometry is what stays fixed per round so a submission's score means
"how good is this policy", not "which round did it land on".

**Where this was pushed back on, and the resolution.** Asked directly whether box count should
instead be a *variable* — e.g. for difficulty tuning — which is a fair question this design didn't
originally answer. The honest position: difficulty progression and evaluation-variance control are
two different needs, and conflating them by randomising count-per-round would fix neither cleanly
(it adds noise to the metric without giving any operator control over WHEN the course gets
harder). Decision: **box count is a competition-VERSION constant, not a per-round random draw or a
round-input knob.** If the field saturates (miners converge on 20 boxes the way upstream's on-ramp
eventually will get walked), the escalation path is a new spec `version` with a different
`N_SCRAMBLE`/`N_PUSH`/`N_CLIMB` in `env/course.py` — the same pattern upstream used for its own
one-way-door decisions (arms; friction randomisation). This was deliberately NOT built as a
declared-but-non-randomised round-input field (e.g. `difficulty: "standard"|"dense"`) despite that
being a live, defensible alternative that preserves determinism per tier — rejected for now only
because there is no evidence yet that miners need more than one difficulty point, and shipping
tiers before that evidence exists fragments the leaderboard's statistical basis (each tier would
need its own evaluation-sizing pass, HANDOFF.md SS4) for a need that hasn't been demonstrated.
**Revisit this rejection once round-over-round score data shows convergence** — the tiered-input
approach remains the right next move if/when that happens, not a version bump, because at that
point the operator wants a lever without a release cycle.

## Zone design: dash / scramble / push / climb

The brief named four verbs the crossing should force. Each maps to a length of the room and a
distinct sampling regime, west to east along +x:

| Zone | Extent | Boxes | Verb forced | How |
|---|---|---|---|---|
| start apron | 0.0-2.0 m | 0 | (settle into gait) | clear floor, mirrors upstream's own run-up |
| scramble field | 2.0-5.0 m | 9 | scramble | small/medium boxes in two staggered, offset rows spanning the full width — no straight lane exists in any of the three lanes without meeting a box |
| push corridor | 5.0-8.0 m | 5 | push | low-density boxes placed AT lane centre, one per ~0.6 m of length, alternately offset so consecutive boxes don't align into one wall — displacing one is the intended solution, not a scripted rule |
| climb stack | 8.0-11.0 m | 6 | climb | high-density, broad-and-squat boxes stacked 2-3 tiers per pile (2 side piles + 1 centre pile), tier tops landing at 0.6-1.3 m — above upstream's proven-feasible 0.55 m single-leg step-up, so only a genuine multi-mount climb clears it |
| dash finish | 11.0-12.0 m | 0 | dash | clear straight sprint to the line, mirrors the scramble zone's opposite |

This is the room-crossing analogue of upstream's own principle: progress-based scoring wants a
continuous difficulty gradient along the crossing, not discrete pass/fail tiers, so the *order* of
zones (not just their presence) matters — scramble first (near-start, recoverable if botched),
push second (need momentum/balance from having cleared the field), climb last (highest skill
floor, closest to the goal so a near-miss still scores well via partial progress).

## Push corridor sizing: what makes a box "pushable"

A legs-only G1 (32.1 kg, no arms — see upstream's own audit) cannot grasp; the only way to move a
box is a sustained body-check, like a person shoulder-checking a light crate while walking through
it rather than picking it up. Density band (25-90 kg/m^3) is set against real-world anchors —
this is drier and lighter than packed cardboard (150-350) — deliberately, so a box in this band at
the sampled size range (0.35-0.80 m side, 0.18-0.42 m tall) has mass in the single-digit-to-low-
tens of kg: light enough that MuJoCo's contact solver visibly displaces it under sustained forward
pelvis-and-torso contact force at G1-plausible walking momentum, without needing a scripted "push"
action or a special contact rule. This was NOT validated against a real trained policy the way
upstream calibrated its hurdle/step-up/duck-bar against the stock walker — flagged as required
pre-onboarding work (see "Open" below): the density band is a physically-reasoned estimate, not a
measured one, exactly the gap upstream's own on-ramp section warns against ("difficulty was set by
driving the stock walker over candidate geometry" — this fork has not yet done that for its push
band because there is no trained-for-this-course baseline yet to drive it with).

## Climb stack sizing: what makes a pile good footing

Two separate properties, both handled through the same density parameter deliberately (rather than
adding a second free variable): **stability under a standing load** (won't slide/tip when a 32 kg
robot commits weight to it — needs both mass AND the friction that upstream's own friction-by-
surface-kind logic ties to material, hence `_friction_band` scaling friction with density in
`env/course.py`) and **stack self-support under gravity alone** (won't topple from its own weight
before the robot ever arrives — handled geometrically, not by density: `_sample_climb_boxes`
narrows each tier's footprint by 12% over the one below, so a pile pyramids the way a real box
stack does, rather than depending on a lucky topple check). Tier heights (0.26-0.46 m each,
stacking to 0.6-1.3 m for 2-3 tiers) are set explicitly ABOVE upstream's measured single-leg
step-up ceiling (0.55 m, needs ~31-63 N.m at the knee against a 139 N.m limit) so a stack cannot
be cleared by the same one-motion mechanic that clears upstream's step-up obstacle — it has to be
a genuine sequential climb (mount tier 1, stand, mount tier 2, ...). This audit follows upstream's
own stated principle exactly: "obstacle sizing must be audited against LEG capability, not against
a robot with hands" (docs/design.md, inherited) — the climb zone's whole design is proving that a
legs-only robot's climb ceiling is lower than its step-up ceiling, and building obstacles that sit
in that gap on purpose.

## What the policy can and cannot see

The height scan (9x5 grid, unchanged mechanism from upstream) hits WHATEVER is directly below each
ray — floor or a box top, whichever is higher, exactly like a real height-map/depth sensor would.
There is no separate "this is a box" or "this is zone: climb" channel. This is a deliberate
simplification, not an oversight: an unlabelled height field is already sufficient to express all
four verbs (a dash lane reads as flat; a scramble cluster reads as broken, closely-spaced bumps; a
push box reads as a step the height scan says is there but which moves when contacted; a climb
stack reads as a tall step with another tall step behind it) without leaking which zone a policy
is in — a policy that could read "zone: push" directly would be solving a different, easier problem
(zone-conditioned behaviour switching) than the one this course means to pose (perceive geometry,
decide whether to detour/push/climb from the geometry alone). `SCAN_CLIP` was raised from
upstream's 1.0 m to 1.5 m specifically so the scan has headroom to report a 1.3 m stack top
relative to a pelvis still standing on the 0.8 m floor below it — without this the scan would
saturate before a climb stack's true height was distinguishable from a shorter obstacle.

## What changed in the cost model (and what has NOT yet been re-measured)

Upstream's careful compile-once optimisation ("The scene is compiled once, not per instance")
assumed geometry never changes within a competition's lifetime and only friction (a *runtime*
MuJoCo field) varies between rounds. This fork breaks that assumption on purpose: the box field
is compile-time geometry (MJCF body/geom pos and size), sampled fresh per ROUND from the round
seed. The adaptation (`env/sim._round_scene`, keyed by round seed rather than a single global
model) keeps the SAME optimisation principle at one level up: compile once per round, reuse
across every instance in that round — a single referee process only evaluates one round, so in
practice this is exactly as cheap as upstream's single compiled model, just re-derived from a
different cache key.

**What has NOT been validated and is real, stated risk carried into HANDOFF.md**: boxes are free
bodies with active contact constraints (up to 20 simultaneously, more when several are touching
in a scramble cluster or a climb stack), which is categorically more expensive per physics step
than upstream's all-static-geometry course. `spec.yaml`'s `evaluate.timeout_s` / `referee.timeout_s`
and the input schema's `max_steps_per_episode` ceiling are carried over from upstream's measured
numbers as a STARTING PRIOR, explicitly flagged inline as unmeasured for this fork. Before
onboarding: profile a full 24-instance suite under the spec's own resource limits the way upstream
did (docs/design.md "The scene is compiled once, not per instance" is the template for how to do
this), and adjust `max_steps_per_episode` / `num_instances` / timeouts to fit, exactly as upstream's
own 15 MB / 3000-step sizing exercise did for submission size against inference cost.

## Rejected

- **Sampling box count per round.** See "Box count: why 20, and why fixed" above — the full
  argument and the recorded pushback are there rather than duplicated here.
- **A dedicated "box present" or "zone identity" observation channel.** Considered so a policy
  could plan a global route from the field layout in one shot. Rejected because it would let a
  policy solve "which zone am I in, switch strategy" instead of "what does the terrain in front of
  me look like, decide what to do about it" — the latter is the actual skill upstream's own height
  scan is designed to test, just applied to a richer field. See "What the policy can and cannot
  see" above.
- **Declared (non-randomised) difficulty tiers as a round-input field.** A real alternative to the
  version-bump escalation path chosen for box count; not built because there is no evidence yet
  that miners need more than one difficulty point. See "Box count" above for the full argument and
  the condition under which this should be revisited.
- **Box-count-conditioned or per-zone baseline_raw_score.** Considered so the entry bar could track
  a difficulty knob if one existed. Moot while count is a version constant, not a round input —
  revisit only if the round-input tier alternative above is ever adopted.

## Open

1. **Push-band and climb-band sizing have not been driven against a real trained policy.**
   Upstream calibrated every obstacle (on-ramp angle, hurdle height, step-up height, duck-bar
   height) against the stock G1 walker's actual measured limits before shipping. This fork's push
   density band (25-90 kg/m^3) and climb tier heights (0.26-0.46 m) are reasoned from physical
   plausibility and upstream's own leg-capability audit, not measured against a policy that has
   actually tried to push or climb in this exact scene. **Required before onboarding**: repeat
   upstream's methodology — drive a stock/naive walker (or a lightly fine-tuned one) through each
   zone, confirm the push band is neither "too heavy to ever move" nor "moves at a shove" (trivial),
   and confirm the climb tiers are neither unclimbable by any embodiment nor step-up-able in one
   motion.
2. **Evaluation cost under box contact dynamics has not been profiled.** See "What changed in the
   cost model" above. Blocks setting real (not inherited-placeholder) values for
   `evaluate.timeout_s`, `referee.timeout_s`, and the input schema's `max_steps_per_episode`
   ceiling.
3. **Score variance (sigma_round) for this course has not been measured.** Upstream's own sizing
   procedure (measure sigma_round across >=20 seeds with a real policy, check it against 1/4 of
   the 1% takeover margin) applies here too and has not been run — required before finalising
   `num_instances` in HANDOFF.md SS4. Box contact outcomes (does a push connect solidly or glance
   off; does a climb mount succeed on the first attempt) are plausibly a source of MORE
   per-instance variance than upstream's friction/wind alone, which argues for measuring rather
   than assuming the inherited N=24 is sufficient.
4. **baseline_raw_score is unset (0.0 placeholder).** No baseline policy has been built for this
   course yet — unlike upstream, which ships Unitree's stock walker wrapped to the interface, this
   fork has no reference policy that has ever attempted the box field. `tools/make_baseline.py` is
   carried over unmodified and produces upstream's flat-ground baseline verbatim; it will almost
   certainly stall at the scramble field's first cluster (the stock walker has no terrain
   perception at all, per upstream's own README), which is fine as an honest 0.0-ish starting
   score but should be run and recorded in `baseline/PROVENANCE.md` rather than left unmeasured.

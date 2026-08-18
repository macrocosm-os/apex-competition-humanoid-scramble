<!-- render stills with `python tools/preview.py` and drop the overview image link here, mirroring
     upstream humanoid-parkour's README -->

# Box Scramble

An Apex competition (Bittensor Subnet 1). Fork of
[Humanoid Parkour](https://github.com/macrocosm-os/apex-competition-humanoid-parkour). Miners
submit an **ONNX policy** that drives a Unitree G1 humanoid across a 48 m x 6 m room scattered
with a per-round-sampled field of 196 loose boxes: a dense scramble cluster, a corridor of light
shovable crates, and stacks of heavy boxes too tall to step onto directly.

| | |
|---|---|
| id / version | `box_scramble` 0.1.0 |
| robot | Unitree G1, **22 actuated DoF** — 12 legs + 10 arms (shoulder pitch/roll/yaw, elbow, wrist roll x2 sides), full arm control (2026-08-18, was 12 leg-only DoF) |
| submission | ONNX graph, ≤ 15 MB, architecture free (same interface as upstream) |
| interface | `obs[136]` + `state_in[256]` → `action[22]` + `state_out[256]`, float32 (was `obs[104]`/`action[12]` — BREAKING change, 2026-08-18) |
| evaluation | 24 instances (inherited from upstream, **not yet re-measured for this course** — see docs/design.md), ≤ 3000 control steps each, box field + wind drawn per round |
| baseline | **not yet built** — see docs/design.md, "Open" |

## The robot has full arm control (2026-08-18)

**Changed from the original fork.** The first version of this course kept upstream's legs-only
G1 (12 actuated DoF, arms welded as dead collision geometry) on purpose, matching upstream's own
"push = body-check, climb = leg-mounts only" design. Crux explicitly asked for that reversed:
*"replace the base model with one that has full control of its arms—this is important as there
will be pushing, lifting and climbing involved."* The robot now has 10 actuated arm DoF (5 per
arm: shoulder pitch/roll/yaw, elbow, wrist roll) on top of the unchanged 12 leg DoF —
**22 actuated DoF total**, up from 12. See `env/assets/g1_22dof.xml` and `env/sim.py`'s module
docstring for the full spec (joint ranges/torque limits taken from Unitree's own published
29-DoF G1 model, adapted to this repo's vendored meshes) and what changed as a result (action/
observation dims, PD gains, default pose — all BREAKING changes to the interface; a submission
built for the old 12-DoF/104-obs contract will not load here).

This is a genuine one-way-door reversal, not a tweak: `docs/design.md` still carries the ORIGINAL
design rationale for the legs-only decision as a historical record (it explains real tradeoffs
that mattered when the course had no arms), immediately followed by the 2026-08-18 update
explaining why it was reversed. Read both, not just the newest note, if you want the full
reasoning trail on why arms were added and what it costs (see "Open" below — arm PD gains and
the new hand-proximity observation channel are unvalidated against a real trained policy, same
category of gap the original push/climb box-band sizing already carried).

## The room

48 m long, 6 m wide (length doubled twice from the original 12 m brief; width unchanged — see
docs/design.md for why the room is no longer 2:1). West to east:

| Zone | Extent | Boxes | Forces |
|---|---|---|---|
| start apron | 0.0 – 8.0 m | 0 | settle into gait |
| **mixed field** (scramble + push, interleaved) | 8.0 – 45.0 m | 100 scramble / 60 push | weaving and displacing light boxes, spread across the whole field, not confined to a sub-zone |
| **climb zone** (second half only) | 24.0 – 45.0 m | 36 | a genuine multi-mount climb — 1–2 tiers per pile (not always two), tops above the single-leg step-up ceiling; boxes up to 30% bigger than the original band |
| dash finish | 45.0 – 48.0 m | 0 | short sprint to the line |

**196 boxes total, always** — a fixed number split by role (100/60/36, the original 9/5/6 ratio
scaled with the room), not itself randomised per round. What varies round to round is each box's
size, density, and placement jitter, drawn from the round seed (`env/course.sample_boxes`), with a
real no-overlap placement pass (rejection sampling against a shared footprint registry) so boxes
don't spawn interpenetrating. See docs/design.md for the packing-fraction math behind the
room-size-to-box-count ratio, and for the explicit reasoning on why count is fixed rather than
sampled (short version: sampling count per round would blend policy skill with luck of the draw
into one noisy number — the same anti-Goodhart argument upstream makes for keeping its own course
geometry fixed and randomising only friction/wind).

Boxes are physical bodies with mass derived from sampled density — pushing and climbing are real
contact-solver outcomes, not scripted animations.

| Zone | Box side (m) | Height (m) | Density (kg/m³) |
|---|---|---|---|
| scramble | 0.28 – 0.85 | 0.22 – 0.60 | 60 – 390 (light clutter) |
| push | 0.35 – 0.80 | 0.18 – 0.42 | 37.5 – 135 (deliberately shovable) |
| climb | 0.675 – 1.95 per tier | 0.39 – 0.897 per tier | 525 – 2100 (stable footing) |

```bash
python -m env.course --seed 1     # print one round's box layout
python tools/preview.py --seed 1  # stills + flythrough (needs mujoco + ffmpeg)
```

## Scoring

Identical to upstream, unchanged:

| Outcome | Score |
|---|---|
| completed | `1.0 + (max_steps - steps) / max_steps` → (1.0, 2.0] |
| fell / timeout / out_of_bounds | `progress`, the fraction of the room crossed → [0.0, 1.0) |
| physics_glitch / invalid / player error | 0.0 |

`raw_score` is the mean over the instances. Progress is continuous along the room regardless of
which zone a robot is in, so a policy that gets 2 m further into the scramble field scores 2 m
better even without clearing it — the same continuous-gradient principle upstream uses.

## Why the box field is randomised, not fixed

Same reasoning as upstream's own friction/wind randomisation, applied one level up: if the box
field were public and static, the cheapest route to the top would be solving one known layout
offline and replaying the trajectory, rather than learning to perceive and react to terrain. The
whole field (every box's size, density, and position) is drawn from a per-round seed
(`env/course.sample_boxes`) that is not published while the round is open. **Unlike** upstream,
this fork's "what's fixed" list is shorter — upstream keeps geometry public and fixed forever,
randomising only friction and wind; this course keeps the room's *shape* (dimensions, zone
lengths, box count and roles) fixed across every round, but the field's specific instantiation
(which sizes, which densities, which exact positions) rotates every round along with wind.

## Perception

Same channels as upstream: proprioception, pose on the track, a height scan (9×5 grid, 0.4 m
behind to 1.6 m ahead) and overhead/forward clearance (7 samples ahead). The height scan reports
whatever is directly below each ray — floor or a box top, whichever is higher — with no separate
"this is a box" or "this is zone X" channel. A stack simply reads as a tall step; a scramble
cluster reads as broken, closely-spaced bumps. See docs/design.md, "What the policy can and cannot
see", for why that's a deliberate design choice and not a missing feature.

## Submitting

Interface is now `obs[136]` + `state_in[256]` → `action[22]` + `state_out[256]` (2026-08-18: was
byte-for-byte identical to upstream Humanoid Parkour at `obs[104]`/`action[12]`; the arm-control
change above is a deliberate BREAKING change to that parity). Same 15 MB ONNX cap, same
recurrent-state contract (feed-forward policies simply return zeros for `state_out`). A
submission tuned for upstream's legs-only course, or for this course's own pre-arms interface,
will NOT load here — the tensor shapes no longer match and the player's readiness check rejects
it as a typed submission failure, not a silent truncation.

## Status

**This spec is a design draft, not yet onboarded.** Before it can go to
[Competition onboarding](https://github.com/macrocosm-os/apex-competitions-builder/issues/new?template=competition-onboarding.yml),
the following from `docs/design.md`'s "Open" section still need doing:

1. Player + referee images built, cosign-signed, and pushed by digest (spec.yaml currently carries placeholder digests).
2. Push-band and climb-band sizing driven against a real (or lightly fine-tuned) policy, the same way upstream calibrated its hurdle/step-up/duck-bar against the stock walker.
3. Evaluation wall-clock re-profiled under box contact dynamics — free-body physics is more expensive per step than upstream's static geometry, and `evaluate.timeout_s`/`referee.timeout_s`/`max_steps_per_episode` are currently inherited placeholders.
4. Score variance (σ_round) measured across ≥20 seeds per `reference/evaluation-design.md`'s sizing procedure, to confirm or revise `num_instances` (currently inherited at 24).
5. A baseline policy built and scored end-to-end, with its provenance recorded in `baseline/PROVENANCE.md` (currently unset — `defaults.baseline_raw_score: 0.0` is a placeholder, not a measurement).

## Repo layout

```
env/            room + box-field sampling, physics, perception, gates, scoring, history format
  course.py     FORKED — the room and box field (was: the linear maneuver sequence)
  sim.py        FORKED — round-scoped scene compilation, box-aware ray casts (was: course-scoped)
  scoring.py    unchanged from upstream
  history.py    forked — records round_seed instead of per-geom frictions
  assets/       vendored Unitree G1 12-DoF model + collision meshes (BSD-3), unchanged
player/         ONNX serving + interface validation (player image), unchanged from upstream
referee/        forked — match driver reads the new env/ modules; no friction/mu in metadata
baseline/       PROVENANCE.md carried from upstream; NOT yet re-run against this course (see Status)
tools/          forked — preview/replay rebuild scenes from round seed, not friction array
docs/           design notes, including the box-count-fixed-vs-variable decision and open items
spec.yaml       the competition manifest — placeholder image digests, inherited timeout/sizing
```

## Provenance

Built as a fork of
[macrocosm-os/apex-competition-humanoid-parkour](https://github.com/macrocosm-os/apex-competition-humanoid-parkour),
itself built from
[apex-competition-hello-world](https://github.com/macrocosm-os/apex-competition-hello-world). The
robot model, `gym_v1` vendoring, player serving logic, and scoring formula are carried over
unmodified; the room, box field, and their sampling/physics are new.

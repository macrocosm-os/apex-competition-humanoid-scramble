<!-- render stills with `python tools/preview.py` and drop the overview image link here, mirroring
     upstream humanoid-parkour's README -->

# Box Scramble

An Apex competition (Bittensor Subnet 1). Fork of
[Humanoid Parkour](https://github.com/macrocosm-os/apex-competition-humanoid-parkour). Miners
submit an **ONNX policy** that drives a Unitree G1 humanoid across a 12 m x 6 m room scattered
with a per-round-sampled field of 20 loose boxes: a dense scramble cluster, a corridor of light
shovable crates, and a stack of heavy boxes too tall to step onto directly.

| | |
|---|---|
| id / version | `box_scramble` 0.1.0 |
| robot | Unitree G1, **12 actuated leg DoF only** — no arm joints, 32.1 kg (unchanged from upstream) |
| submission | ONNX graph, ≤ 15 MB, architecture free (same interface as upstream) |
| interface | `obs[104]` + `state_in[256]` → `action[12]` + `state_out[256]`, float32 |
| evaluation | 24 instances (inherited from upstream, **not yet re-measured for this course** — see docs/design.md), ≤ 3000 control steps each, box field + wind drawn per round |
| baseline | **not yet built** — see docs/design.md, "Open" |

## The robot has no arms

Unchanged from upstream: all 12 actuators are legs. The arms are 17.7 kg of collision geometry
welded to the pelvis — present, with mass, and they hit things, but nothing can move them. That
means **pushing a box is a body-check, not a shove with hands**, and **climbing a stack is a
sequence of leg mounts, not a pull-up**. See docs/design.md, "Push corridor sizing" and "Climb
stack sizing", for exactly how the box field is calibrated against that constraint.

## The room

12 m long, 6 m wide (aspect 2:1, per the brief this fork was built against). West to east:

| Zone | Extent | Boxes | Forces |
|---|---|---|---|
| start apron | 0.0 – 2.0 m | 0 | settle into gait |
| **scramble field** | 2.0 – 5.0 m | 9 | weaving — two staggered rows spanning the full width; no lane has a clean straight line through |
| **push corridor** | 5.0 – 8.0 m | 5 | displacing light boxes placed directly in the lane, or a very tight detour |
| **climb stack** | 8.0 – 11.0 m | 6 | a genuine multi-mount climb — piles are 2–3 tiers, tops at 0.6–1.3 m, above the single-leg step-up ceiling |
| dash finish | 11.0 – 12.0 m | 0 | sprint to the line |

**20 boxes total, always** — a fixed number split by role (9/5/6), not itself randomised per
round. What varies round to round is each box's size, density, and placement jitter, drawn from
the round seed (`env/course.sample_boxes`). See docs/design.md for the packing-fraction math
behind the room-size-to-box-count ratio, and for the explicit reasoning on why count is fixed
rather than sampled (short version: sampling count per round would blend policy skill with luck
of the draw into one noisy number — the same anti-Goodhart argument upstream makes for keeping its
own course geometry fixed and randomising only friction/wind).

Boxes are physical bodies with mass derived from sampled density — pushing and climbing are real
contact-solver outcomes, not scripted animations.

| Zone | Box side (m) | Height (m) | Density (kg/m³) |
|---|---|---|---|
| scramble | 0.28 – 0.85 | 0.22 – 0.60 | 40 – 260 (light clutter) |
| push | 0.35 – 0.80 | 0.18 – 0.42 | 25 – 90 (deliberately shovable) |
| climb | 0.45 – 1.00 per tier | 0.26 – 0.46 per tier | 350 – 1400 (stable footing) |

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

Interface is byte-for-byte identical to upstream Humanoid Parkour: `obs[104]` + `state_in[256]` →
`action[12]` + `state_out[256]`, same 15 MB ONNX cap, same recurrent-state contract (feed-forward
policies simply return zeros for `state_out`). If you have a submission tuned for upstream's
linear course, it will load here without modification — it almost certainly will not get very
far, because it has never seen a box.

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

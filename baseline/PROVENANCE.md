# baseline.onnx

The number in `spec.defaults.baseline_raw_score` is what this artifact scores. It is the bar a
submission has to clear by 1% to take the competition over, so where it comes from matters.

## What it is

Unitree's stock G1 walking policy, `deploy/pre_train/g1/motion.pt` from
[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) (BSD-3), wrapped to speak
this competition's interface. It is **not trained on this course** and cannot see it — the
wrapper feeds it none of the perception channels. It is a flat-ground locomotion policy that
happens to be walking on a parkour track.

Architecture: `LSTM(47 -> 64)` then `Linear(64 -> 32) -> ELU -> Linear(32 -> 12)`. 0.13 MB.

`tools/make_baseline.py` builds it and is the authority on the details. Three things happen there:

1. **Observation slicing.** The competition's 104-d observation is cut down to the 47-d vector
   motion.pt was trained on. The scaling factors already match, by construction: `env/sim.py`
   scales angular velocity by 0.25 and joint velocity by 0.05 because those are Unitree's.
2. **Command synthesis.** motion.pt tracks a *body-frame* velocity command with no heading
   feedback, so it needs one supplied. The wrapper closes a proportional heading-hold loop on the
   course centreline. Without it the policy drifts ~0.26 m sideways per metre travelled and walks
   off the track before reaching any obstacle — which would have made the course look far harder
   than it is.
3. **State threading.** motion.pt is recurrent, and its TorchScript build hides the LSTM state in
   mutating buffers that neither ONNX exporter will trace. The weights are lifted into a plain
   `nn.Module` that threads state through the competition's `state_in`/`state_out` tensors.

That last step is a re-implementation, so it is checked rather than assumed:
`make_baseline.py --check` rolls both the original and the rebuild over 64 random observations
and asserts the actions agree. **Measured max |delta| = 0.0e+00** — bit-identical.

## What it scores

Measured **inside the referee image, on a native amd64 runner** over the full 24-instance suite
at `fixtures/input.json` (`num_instances: 24`, `max_steps_per_episode: 4000`, `deadline_ms: 500`)
by `.github/workflows/measure-baseline.yml`:

```
raw_score        0.20068353334086175
instances        24
completed        0
furthest         10.73 m of 51.1 m
terminal reasons fell x 24
eval time        66.8 s   (referee timeout 900 s; ~258 s worst case)
```

Where it is measured matters more than it looks, because all three numbers differ inside the
1% takeover margin:

| measured | raw_score | delta |
|---|---|---|
| referee image, native amd64 — **the spec figure** | 0.20068 | — |
| host (arm64, no container) | 0.2006 | 0.04% |
| referee image, arm64 build | 0.20044 | 0.12% |

So it is pinned to amd64-in-image, because that is what the platform runs. It cannot be measured
on an arm64 dev machine at all: published images are amd64-only, and under qemu emulation a
24-instance suite did not finish in 15 hours.

Within one image on one architecture it is exact. On the arm64 build, four distinct `SEED` values
(12345, 777, 999888, and one random) all reproduce `0.2004409785` **bit for bit**. That is the point of the fixed
evaluation suite (`env/sim.instance_spec`): round-to-round variance is zero, so takeover is
decided by skill rather than by which instances a round happened to draw.

## Where it dies

Every instance ends the same way: the policy walks the 6 m run-up, climbs the 15.4° on-ramp, and
falls at the 0.55 m drop on the far side. It has no landing controller, because nothing on flat
ground ever asked it for one.

So the honest reading of `0.2007` is **"a competent off-the-shelf locomotion policy clears the
first 21% of this course and then falls off a ledge."** Everything past the on-ramp — the stairs,
the 1 m leap, the hurdle, the step-up, the duck-under, the beam, the slick patch — is unscored
territory. No policy has completed the course.

## Reproducing

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_rl_gym
cd unitree_rl_gym && git sparse-checkout set resources/robots/g1_description \
    deploy/pre_train/g1 deploy/deploy_mujoco && cd ..

python tools/make_baseline.py --urlg unitree_rl_gym     # writes baseline/baseline.onnx
python tools/local_eval.py baseline/baseline.onnx -n 24 # host-side score
```

For the in-image number, build the referee and player images and run a match against them — see
the end-to-end invocation in the README.

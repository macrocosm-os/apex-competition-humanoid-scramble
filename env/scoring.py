"""Per-instance scoring for Box Scramble. Unchanged from upstream Humanoid Parkour (see
docs/design.md, "Rejected -- Checkpoint scoring"): progress along the room already gives a
smooth gradient regardless of which zone (scramble/push/climb) is being crossed, so no
per-zone bonus or checkpoint is needed. Shared by the referee and the local eval /
variance-measurement tools so the numbers can never diverge.

Per room instance (higher is better):
    completed        1.0 + (max_steps - steps) / max_steps   -> in (1.0, 2.0]
    fell / timeout / out_of_bounds / time_limit
                     progress (fraction of room crossed)   -> in [0.0, 1.0)
    physics_glitch / invalid or errored player
                     0.0

A run scores for the room it crossed under its own physics. Anything the SUBMISSION did wrong -- a
state outside the physical regime, an action that is not one, a policy that stopped answering --
is not a crossing of the room and scores nothing.

`time_limit` is deliberately NOT in that list, changed 2026-09-08 after the PR-env load test
(apex-mvp#452). It is the only terminal reason the submission does not cause: it means the
referee's wall-clock budget ran out, which depends on how many other evaluations happened to share
the node. Scoring it zero made the platform's contention change what a submission scored, and
made it worse the better the submission was -- a policy that survives runs longer, so it eats more
clock, so it is the one that gets cut. Measured over 49 runs of one identical artifact on one
round seed: 43.7% of instances ended `time_limit`, the cut instances had travelled FURTHER
(11.4-11.6 m) than the ones that ended naturally (8.0-11.5 m), and the suite returned 19 distinct
scores spanning 0.000-0.181 where it should have returned one number.

Scoring the progress it actually made removes the anti-correlation and leaves contention affecting
only how much of the room a run gets to attempt, not whether the part it crossed counts. It also
cannot be gamed: the budget is shared across the suite, so burning clock on one instance starves
the rest and lowers the mean. Instances the budget never reached still score 0.0 -- they have zero
progress -- so nothing is credited to a run that never happened.

Any completion outranks any non-completion, faster completions outrank slower
ones, and partial progress gives non-completing miners a training gradient.
The round raw_score is the mean over all room instances.
"""

from __future__ import annotations


def instance_score(terminal_reason: str, progress: float, steps: int, max_steps: int) -> float:
    if terminal_reason == "completed":
        return 1.0 + (max_steps - steps) / max_steps
    if terminal_reason in ("fell", "timeout", "out_of_bounds", "time_limit"):
        return progress
    # physics_glitch, invalid_action, player_error: typed zero.
    return 0.0

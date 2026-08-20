"""Per-instance scoring for Box Scramble. Unchanged from upstream Humanoid Parkour (see
docs/design.md, "Rejected -- Checkpoint scoring"): progress along the room already gives a
smooth gradient regardless of which zone (scramble/push/climb) is being crossed, so no
per-zone bonus or checkpoint is needed. Shared by the referee and the local eval /
variance-measurement tools so the numbers can never diverge.

Per room instance (higher is better):
    completed        1.0 + (max_steps - steps) / max_steps   -> in (1.0, 2.0]
    fell / timeout / out_of_bounds
                     progress (fraction of room crossed)   -> in [0.0, 1.0)
    physics_glitch / time_limit / invalid or errored player
                     0.0

A run scores for the room it crossed under its own physics, within the step cap and the
evaluation's time budget. Anything else -- a state outside the physical regime, a run that used up
the clock, an action that is not one, a policy that stopped answering -- is not a crossing of the
room and scores nothing.

Any completion outranks any non-completion, faster completions outrank slower
ones, and partial progress gives non-completing miners a training gradient.
The round raw_score is the mean over all room instances.
"""

from __future__ import annotations


def instance_score(terminal_reason: str, progress: float, steps: int, max_steps: int) -> float:
    if terminal_reason == "completed":
        return 1.0 + (max_steps - steps) / max_steps
    if terminal_reason in ("fell", "timeout", "out_of_bounds"):
        return progress
    # physics_glitch, time_limit, invalid_action, player_error: typed zero.
    return 0.0

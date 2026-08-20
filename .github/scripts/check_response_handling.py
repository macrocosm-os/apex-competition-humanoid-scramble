"""Everything a player returns is attacker-controlled input to the scorer.

Each response below must come back as a scored instance, and none of them may raise out of
play_game: an escaping exception is attributed to the REFEREE, not to the submission that caused
it, which stalls the round. The suite's wall-clock budget is checked the same way — a policy that
answers legally but slowly must not be able to run the referee past its own timeout.

    PYTHONPATH=.:referee:player python .github/scripts/check_response_handling.py
"""

from __future__ import annotations

import time

import referee as ref                      # referee/referee.py
from gym_v1.referee import RefereeContext

from env.sim import ACT_DIM

CONFORMING = [0.0] * ACT_DIM

# Named cases, all of them things a submission can put on the wire. The expected terminal reason
# is deliberately not asserted per case — what matters is that the instance is SCORED at the zero
# floor and that nothing raises.
CASES = {
    "oversized-nested": [[0.0] * 200_000] * 22,
    "over-long": [0.0] * 5000,
    "too-deep": [[[0.0] * ACT_DIM]],
    "wide-and-nested": [[0.0] * 8] * 700,
    "nan": [float("nan")] * ACT_DIM,
    "inf": [float("inf")] * ACT_DIM,
    "text": "forward",
    "text-in-list": ["x"] * ACT_DIM,
    "dict": {"action": CONFORMING},
    "empty": [],
    "none": None,
    "short": [0.0] * (ACT_DIM - 1),
}


class Stub:
    """A player that answers whatever it was told to, optionally slowly."""

    SENTINEL = object()

    def __init__(self, action=SENTINEL, per_call_s: float = 0.0):
        self.action, self.per_call_s = action, per_call_s

    def reset(self, **kw):
        pass

    def act(self, observation, deadline_ms):
        if self.per_call_s:
            time.sleep(self.per_call_s)
        return CONFORMING if self.action is Stub.SENTINEL else self.action


def play(player, instances=1, steps=400, budget=120.0):
    ctx = RefereeContext(match_id="ci", seed=3, num_players=1, player_urls=["http://stub"],
                         config={"seed": 3, "num_instances": instances,
                                 "max_steps_per_episode": steps, "deadline_ms": 500,
                                 "record_history": False, "time_budget_s": budget})
    return ref.BoxScrambleReferee().play_game(ctx, [player])


for name, action in CASES.items():
    result = play(Stub(action=action))
    row = result.metadata["instances"][0]
    assert len(result.metadata["instances"]) == 1, result.metadata
    assert result.raw_scores[0] == 0.0, (name, result.raw_scores)
    print(f"ok: {name:17s} -> {row['terminal_reason']:14s} score {row['score']}")

# A policy that answers legally but slowly is bounded by the suite's own clock, not by the
# platform's hard kill: every instance is still scored and still in the denominator.
N, BUDGET = 4, 1.0
start = time.monotonic()
result = play(Stub(per_call_s=0.5), instances=N, steps=3000, budget=BUDGET)
elapsed = time.monotonic() - start
rows = result.metadata["instances"]
limited = [r for r in rows if r["terminal_reason"] == ref.TIME_LIMIT]
assert len(rows) == N, "instances dropped out of the denominator"
assert limited, "a slow policy never hit the clock"
assert all(r["score"] == 0.0 for r in limited), limited
assert result.raw_scores[0] == sum(r["score"] for r in rows) / N
ceiling = BUDGET + ref.MIN_INSTANCE_BUDGET_S + 5.0
assert elapsed < ceiling, f"suite ran {elapsed:.1f}s against a {BUDGET}s budget"
print(f"ok: slow policy -> {len(limited)}/{N} time_limit in {elapsed:.1f}s "
      f"(budget {BUDGET}s), raw_score {result.raw_scores[0]}")

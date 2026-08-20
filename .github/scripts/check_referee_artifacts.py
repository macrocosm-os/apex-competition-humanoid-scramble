"""The referee's two miner-facing artifacts behave, without needing containers.

Drives the REAL `BoxScrambleReferee.play_game()` against the REAL `ParkourPlayer`: the stub client
below stands in for gym_v1's `PlayerClient`, so every /reset and /act goes through the code the
sandbox runs. That covers both artifacts at once — the per-instance history the referee writes,
and the API log the player prints — and lets the failure path be tested, which a container run
cannot easily do.

    PYTHONPATH=.:referee:player SUBMISSION_PATH=/tmp/test_policy.onnx \
        python .github/scripts/check_referee_artifacts.py
"""

from __future__ import annotations

import contextlib
import io
import pathlib

import referee as ref                      # referee/referee.py
from gym_v1.referee import RefereeContext
from launch import ParkourPlayer           # player/launch.py

from env.history import read_all

N, STEPS = 3, 150

player = ParkourPlayer()
assert player.is_ready(), player.load_error


class Stub:
    """gym_v1's PlayerClient surface — the two methods the referee calls."""

    def reset(self, **kw):
        player.reset(**kw)

    def act(self, observation, deadline_ms):
        return player.act(observation=observation, deadline_ms=deadline_ms)


def play(history_dir: str, instances: int = N, steps: int = STEPS):
    """One scored suite against `history_dir`, returning (result, captured stdout).

    `history_dir` is emptied first: the assertions below count files, so a leftover record from
    an earlier run (or an earlier call here with a different instance count) would fail the check
    for a reason that has nothing to do with the code.
    """
    target = pathlib.Path(history_dir)
    if target.is_dir():
        for stale in target.glob("*.json"):
            stale.unlink()
    ref.HISTORY_DIR = target
    ctx = RefereeContext(match_id="ci", seed=1, num_players=1, player_urls=["http://stub"],
                         config={"seed": 1, "num_instances": instances,
                                 "max_steps_per_episode": steps, "deadline_ms": 500,
                                 # Generous, so a slow shared runner cannot end an instance on
                                 # the clock and make this check flaky. The budget itself is
                                 # exercised by check_response_handling.py.
                                 "time_budget_s": 600})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = ref.BoxScrambleReferee().play_game(ctx, [Stub()])
    return result, buf.getvalue()


# 1. The referee writes one history file per instance, agreeing with what it scored.
result, log = play("/tmp/refhistory")
files = sorted(pathlib.Path("/tmp/refhistory").glob("*.json"))
assert [p.name for p in files] == [f"instance_{i:02d}.json" for i in range(N)], files
assert result.metadata["history_files"] == N, result.metadata
for record, row in zip(read_all("/tmp/refhistory"), result.metadata["instances"]):
    assert record["match_id"] == "ci", record["match_id"]
    assert record["outcome"]["score"] == row["score"], (record["outcome"], row)
print(f"ok: {N} history files, outcomes agree with result.json metadata")

# 2. The player logs every API call the referee made — one act line per control step.
acts = [ln for ln in log.splitlines() if ln.startswith("apex.api") and " call=act " in ln]
resets = [ln for ln in log.splitlines() if " call=reset " in ln]
assert len(resets) == N, len(resets)
assert len(acts) == sum(r["steps"] for r in result.metadata["instances"]), len(acts)
print(f"ok: {len(resets)} reset + {len(acts)} act lines logged (= total control steps)")

# 3. A failure to write a debugging artifact must never turn a scored round into a zero.
ok, _ = play("/tmp/writable_history", instances=2, steps=100)
bad, _ = play("/proc/nope/history", instances=2, steps=100)
assert ok.metadata["history_files"] == 2, ok.metadata      # the control really did write
assert bad.metadata["history_files"] == 0, bad.metadata
assert bad.raw_scores == ok.raw_scores, (bad.raw_scores, ok.raw_scores)
assert [r["score"] for r in bad.metadata["instances"]] == \
       [r["score"] for r in ok.metadata["instances"]]
print(f"ok: unwritable history -> history_files=0, score unchanged at {ok.raw_scores[0]}")

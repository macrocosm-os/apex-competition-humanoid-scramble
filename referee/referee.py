"""box_scramble gym_v1 REFEREE (the scorer sandbox, run at /app/referee.py).

Fork of Humanoid Parkour's referee.py. Owns the physics: builds the evaluation suite, steps
MuJoCo, streams observations to the player over /act, and applies the termination + scoring
gates. The player sandbox only ever sees observation vectors.

raw_score = mean instance score over all instances (see env/scoring.py). Per-instance breakdowns
go in metadata: hidden while the round is active, revealed to miners when it completes.

Alongside the score, each instance writes a replayable history file to /data/history/ — the
platform collects those as FileType.HISTORY, ships them to S3, and lists them on the submission
for the miner to download (see env/history.py, tools/replay.py). A directory is used instead of
one JSONL because the suite runs all instances in ONE referee container, so the per-game unit is
a file rather than a line.

The round seed drives the WHOLE box field (env/course.sample_boxes) once per round, plus each
instance's wind (env/sim.instance_spec) -- so the suite differs every round, but every instance
within one round faces the identical box layout. What the metadata and the history files report is
the conditions each instance faced, not the seed the round was drawn with.

The suite runs under a wall-clock budget as well as a step cap: every instance gets a share of it,
and an instance that runs out of clock ends as `time_limit` and scores as a run that did not cross
the room. The budget is what keeps the suite inside `referee.timeout_s` regardless of how long a
policy takes to answer.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import time

from dataclasses import asdict

from gym_v1 import GameResult, Referee, RefereeContext
from gym_v1.client import PlayerClient, PlayerError
from gym_v1.referee import RESULT_PATH

from env import ParkourSim, instance_score, instance_spec
from env.history import DEFAULT_STRIDE, InstanceRecorder, write_instance
from env.sim import ACT_DIM, FRAME_SKIP, OBS_DIM, PHYS_DT, STATE_DIM, WIND_MAX_MS

# Sized against the referee's 900 s timeout: ~2 s of physics per instance plus HTTP.
# The round input (CONFIG_JSON) can override.
DEFAULT_NUM_INSTANCES = 24
DEFAULT_MAX_STEPS = 3000
DEFAULT_DEADLINE_MS = 500

# Wall-clock the suite is allowed, in seconds, and how it is shared out.
#
# `referee.timeout_s` in spec.yaml is a hard kill with no grace: whatever is unwritten at that
# moment is lost, and the round has no result. So the suite runs to its own budget, comfortably
# inside that, and every instance gets an equal share of what is left when it starts. An instance
# that reaches its share ends as `time_limit` (env/scoring.py scores it as a run that crossed no
# room); once the whole budget is gone the remaining instances end the same way, in the
# denominator, so the score is always the mean over the full suite.
DEFAULT_SUITE_BUDGET_S = 780.0
# Floor on one instance's share, so a large num_instances cannot slice the budget so thin that
# instances end on the clock before physics has had a chance to run.
MIN_INSTANCE_BUDGET_S = 5.0
TIME_LIMIT = "time_limit"

# Sibling of /data/result.json. The worker copies `<mount>/history/*` out of the sandbox before
# wiping it and delivers them as FileType.HISTORY, so the name of this directory is a contract
# with the platform, not a local choice.
HISTORY_DIR = pathlib.Path("/data/history")

# Everything a broken player can throw at us, all of it the SUBMISSION's fault.
#
# The vendored gym_v1 PlayerClient only converts urllib.error.URLError and TimeoutError into
# PlayerError. A player process that dies or raises mid-request surfaces as
# http.client.RemoteDisconnected (a ConnectionResetError, so an OSError) and a player that
# answers with a non-JSON body surfaces as json.JSONDecodeError — neither is a URLError, so
# both escape the client. If they escaped play_game too, the platform would score a bad
# submission as a REFEREE failure, which is the one misattribution the contract forbids.
# RecursionError joins them because it is what json's own scanner raises on a body nested past its
# limit: a response the client could not read, which is the same class of outcome as the rest.
PLAYER_FAULTS = (PlayerError, OSError, json.JSONDecodeError, RecursionError)

# What a well-formed /act response looks like on the wire. An action is ACT_DIM finite numbers,
# so a response is a few hundred bytes; these bounds sit far above that and below anything that
# would be expensive to hold or to walk.
MAX_RESPONSE_BYTES = 1 << 20        # 1 MiB of JSON body
MAX_ACTION_LEN = 1024               # entries in the returned sequence
MAX_ACTION_ELEMENTS = 4096          # entries once nesting is counted
MAX_ACTION_DEPTH = 2                # a flat vector, or a [1, ACT_DIM] batch of one


class BoundedPlayerClient(PlayerClient):
    """`PlayerClient` with the response body bounded.

    The vendored client reads the whole body before parsing it, which is the right default for a
    protocol where responses are a few hundred bytes. This competition's referee holds a MuJoCo
    scene alongside it, so it reads up to `MAX_RESPONSE_BYTES` and treats a longer body the same
    way it treats a malformed one: not a response, and reported as a player fault.
    """

    def _post(self, path: str, body: dict, timeout_s: float):
        import urllib.error
        import urllib.request

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                status = resp.status
                raw = resp.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.URLError as e:
            raise PlayerError(f"POST {path} to {self.base_url} failed: {e}") from e
        except TimeoutError as e:
            raise PlayerError(f"POST {path} to {self.base_url} timed out after {timeout_s}s") from e
        if len(raw) > MAX_RESPONSE_BYTES:
            raise PlayerError(
                f"POST {path} to {self.base_url} answered with more than "
                f"{MAX_RESPONSE_BYTES} bytes"
            )
        payload = json.loads(raw) if raw else {}
        return status, payload


def action_is_well_formed(action) -> bool:
    """True if `action` is small enough and shallow enough to hand to numpy.

    A conforming action is a flat ACT_DIM sequence (or a batch of one). `env.sim.ParkourSim.step`
    is what decides whether the numbers themselves are valid; this only decides whether the
    structure is one worth converting.
    """
    depth, level = 0, [action]
    total = 0
    while level:
        if not isinstance(level[0], (list, tuple)):
            return True
        depth += 1
        if depth > MAX_ACTION_DEPTH:
            return False
        if len(level) > MAX_ACTION_LEN:
            return False
        total += len(level)
        if total > MAX_ACTION_ELEMENTS:
            return False
        nxt = []
        for item in level:
            if isinstance(item, (list, tuple)):
                if len(item) > MAX_ACTION_LEN:
                    return False
                nxt.extend(item)
        if len(nxt) > MAX_ACTION_ELEMENTS:
            return False
        level = nxt
    return True


class BoxScrambleReferee(Referee):
    @staticmethod
    def _unrun_row(i: int, params, reason: str) -> dict:
        """The metadata row for an instance the clock ran out on before it started.

        Same keys as a run instance, so the metadata is one uniform table and the mean is taken
        over the full suite.
        """
        return {
            "instance": i,
            "wind_speed_ms": round(params.wind_speed, 2),
            "wind_dir_deg": round(math.degrees(params.wind_dir), 1),
            "terminal_reason": reason,
            "progress": 0.0,
            "distance_m": 0.0,
            "steps": 0,
            "sim_time_s": 0.0,
            "score": 0.0,
        }

    def play_game(self, ctx: RefereeContext, players: list[PlayerClient]) -> GameResult:
        start = time.monotonic()
        cfg = ctx.config or {}
        n = int(cfg.get("num_instances", DEFAULT_NUM_INSTANCES))
        max_steps = int(cfg.get("max_steps_per_episode", DEFAULT_MAX_STEPS))
        deadline_ms = int(cfg.get("deadline_ms", DEFAULT_DEADLINE_MS))
        wind_max = float(cfg.get("wind_max_ms", WIND_MAX_MS))
        # The round input is authoritative; ctx.seed (platform SEED) is the fallback. Reading both
        # means the draw still rotates if the platform's seed extraction misses the input field.
        seed = int(cfg.get("seed", ctx.seed))
        # On by default, as tron's trace is. The escape hatch exists because history is the one
        # output that grows with the suite, not because the scored path depends on it.
        record_history = bool(cfg.get("record_history", True))
        history_stride = int(cfg.get("history_stride", DEFAULT_STRIDE))
        suite_budget_s = float(cfg.get("time_budget_s", DEFAULT_SUITE_BUDGET_S))
        player = players[0]

        instances = []
        history_written = 0
        total = 0.0
        for i in range(n):
            # Each instance gets an equal share of the clock still standing, so the suite finishes
            # inside its budget whatever the instances before it did with theirs.
            left = suite_budget_s - (time.monotonic() - start)
            instance_budget_s = max(MIN_INSTANCE_BUDGET_S, left / (n - i)) if left > 0 else 0.0
            instance_start = time.monotonic()

            params = instance_spec(i, n, seed, wind_max=wind_max)
            if left <= 0:
                # The budget is spent. The instance still exists and still counts: it is recorded
                # as a run that crossed no room, which is what the mean is taken over.
                instances.append(self._unrun_row(i, params, TIME_LIMIT))
                continue

            sim = ParkourSim(params)
            obs = sim.reset()
            rec = InstanceRecorder(i, sim, history_stride) if record_history else None
            # The box field and wind are what a policy senses (height scan, clearance rays,
            # contact), not something it is told: reset() carries seed=0 and an empty config, so
            # the player sandbox sees nothing that identifies the instance or the round.
            player.reset(match_id=f"{ctx.match_id}:{i}", player_index=0, seed=0, config={})

            reason = None
            while reason is None:
                if time.monotonic() - instance_start > instance_budget_s:
                    reason = TIME_LIMIT
                    break
                # Only the player call is inside the player-fault handler. sim.step() is OUR
                # code: if it raises, that is a referee bug and must surface as a referee
                # failure, not be laundered into a zero for the submission.
                try:
                    action = player.act(observation=obs.tolist(), deadline_ms=deadline_ms)
                except PLAYER_FAULTS:
                    reason = "player_error"  # unreachable / timed out / died / garbage response
                    break
                # An action is a flat vector of ACT_DIM numbers; anything of a different shape or
                # size than that is not one, and is scored as an invalid action rather than
                # converted.
                if not action_is_well_formed(action):
                    reason = "invalid_action"
                    break
                try:
                    result = sim.step(action, max_steps=max_steps)
                except (TypeError, ValueError):
                    # NaN / wrong shape / non-numeric. ValueError covers both `InvalidAction`
                    # (which subclasses it) and numpy's own refusal to read the response as
                    # numbers at all, e.g. text where a float belongs.
                    reason = "invalid_action"
                    break
                obs, reason = result.obs, result.terminal_reason
                if rec is not None:
                    rec.capture(sim, action)

            score = instance_score(reason, sim.progress, sim.steps, max_steps)
            total += score
            instances.append({
                "instance": i,
                "wind_speed_ms": round(params.wind_speed, 2),
                "wind_dir_deg": round(math.degrees(params.wind_dir), 1),
                "terminal_reason": reason,
                "progress": round(sim.progress, 4),
                "distance_m": round(sim.max_x, 2),
                "steps": sim.steps,
                "sim_time_s": round(sim.steps * PHYS_DT * FRAME_SKIP, 2),
                "score": round(score, 4),
            })

            # Best-effort, and deliberately so: a history file is a debugging artifact, and
            # failing to write one must never turn a scored round into a referee failure. The
            # platform's own collection is best-effort for the same reason.
            if rec is not None:
                try:
                    write_instance(HISTORY_DIR, rec.record(
                        sim, instances[-1], match_id=ctx.match_id, num_instances=n))
                    history_written += 1
                except Exception as e:  # noqa: BLE001 — never fail a round over an artifact
                    print(f"history write failed for instance {i}: {type(e).__name__}: {e}",
                          file=sys.stderr, flush=True)
            rec = None   # drop the buffer; peak memory stays one episode, not the whole suite

        completed = sum(c["terminal_reason"] == "completed" for c in instances)
        raw = total / len(instances)
        return GameResult(
            raw_scores=[raw],
            winner=0 if raw > 0 else -1,
            terminal_reason="scored",
            steps=sum(c["steps"] for c in instances),
            metadata={
                "instances": instances,
                "num_instances": len(instances),
                "num_completed": completed,
                "furthest_m": max(c["distance_m"] for c in instances),
                "wind_max_ms": wind_max,
                "history_files": history_written,
                "num_time_limited": sum(c["terminal_reason"] == TIME_LIMIT for c in instances),
                "time_budget_in_seconds": round(suite_budget_s, 1),
                "eval_time_in_seconds": round(time.monotonic() - start, 1),
            },
        )

    def run(self) -> None:
        """Same as the toolkit's Referee.run(), except a player that never becomes ready is
        scored as a typed SUBMISSION failure instead of a referee failure.

        Why this override exists: gym_v1's Referee.run() calls wait_until_ready() BEFORE
        play_game(), so its PlayerError escapes at a point where no /data/result.json can be
        written — and a missing result.json is attributed to the referee. But a player that
        never reports ready is exactly what a malformed ONNX artifact looks like (see
        player/launch.py: a load failure serves is_ready() False rather than dying), which is
        the submission's fault and must come back to the miner as an explained zero.

        This is NOT papering over a referee bug: the scope is one specific PlayerError from the
        readiness wait. play_game() itself is left completely unguarded, so a genuine referee
        crash still produces no result.json and is still attributed to us.
        """
        ctx = RefereeContext.from_env()
        players = [BoundedPlayerClient(url) for url in ctx.player_urls]
        try:
            for p in players:
                p.wait_until_ready(self.readiness_timeout_s)
        except PlayerError as e:
            result = GameResult(
                raw_scores=[0.0],
                winner=-1,
                terminal_reason="submission_not_ready",
                steps=0,
                metadata={
                    "error": str(e),
                    "explanation": (
                        "The submission never became ready. Usually the ONNX artifact failed to "
                        "load or does not match the required interface: inputs obs "
                        f"[batch, {OBS_DIM}] and state_in [batch, {STATE_DIM}], outputs action "
                        f"[batch, {ACT_DIM}] and state_out [batch, {STATE_DIM}], all float32, "
                        "single file with weights embedded, <= 15 MB."
                    ),
                },
            )
        else:
            result = self.play_game(ctx, players)  # unguarded: a crash here is OUR failure

        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(asdict(result)))


if __name__ == "__main__":
    BoxScrambleReferee().run()

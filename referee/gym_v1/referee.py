# VENDORED from macrocosm-os/apex-competitions-builder @ d063d9028dbec4bb15182794496f4aa2aac19d49
# (src/apex_sdk/gym_v1/), import root rewritten apex_sdk.gym_v1 -> gym_v1.
# Do not hand-edit; re-vendor from the toolkit to update.

"""Referee-side harness for gym_v1 (and custom) duels.

The referee image holds the game logic. The platform launches it with the env contract
below, on a per-job network with the player sandboxes. The referee drives the match and
writes the result before exiting.

Env injected by the platform:
    MATCH_ID     opaque string, unique per (job, game_index)
    SEED         int, per-game seed
    CONFIG_JSON  opaque competition config (from the spec / round generator)
    PLAYER_URLS  comma-separated player base URLs, in canonical order
                 (the platform reorders these to implement swap_sides)
    NUM_PLAYERS  int

Output (written before the container exits):
    /data/result.json   {raw_scores, winner, terminal_reason, steps, metadata}
    /data/trace.jsonl    optional, one event per line (shipped to S3 if present)

Failure semantics (enforced by the platform, mirrored here):
    - Player HTTP error/timeout  -> the referee decides (forfeit/retry/draw); raise/catch
      PlayerError in your play_game. The platform does not intervene.
    - Referee crash / no result.json -> the platform scores 0 for all participants and
      attributes the failure to the REFEREE, not the submissions. So we DO NOT write a
      zeroed result on an unexpected crash: we let it propagate (no result.json).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gym_v1.client import PlayerClient

RESULT_PATH = Path("/data/result.json")
TRACE_PATH = Path("/data/trace.jsonl")


@dataclass
class GameResult:
    """The result of one game. Serialized verbatim to /data/result.json."""

    raw_scores: list[float]
    winner: int
    terminal_reason: str
    steps: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RefereeContext:
    """Parsed platform env for one game."""

    match_id: str
    seed: int
    config: dict[str, Any]
    player_urls: list[str]
    num_players: int

    @classmethod
    def from_env(cls) -> "RefereeContext":
        try:
            player_urls = [u for u in os.environ["PLAYER_URLS"].split(",") if u]
            return cls(
                match_id=os.environ["MATCH_ID"],
                seed=int(os.environ["SEED"]),
                config=json.loads(os.environ.get("CONFIG_JSON", "{}")),
                player_urls=player_urls,
                num_players=int(os.environ.get("NUM_PLAYERS", len(player_urls))),
            )
        except KeyError as e:
            raise RuntimeError(f"referee env missing required variable: {e}") from e


class Referee(ABC):
    """Implement play_game with your competition's rules."""

    #: How long to wait for each player to become ready before the game starts.
    readiness_timeout_s: float = 60.0

    @abstractmethod
    def play_game(self, ctx: RefereeContext, players: list[PlayerClient]) -> GameResult:
        """Drive one game to completion and return its result.

        Use players[i].act(...) to query player i. Catch PlayerError to implement your
        own forfeit/draw policy. `self.trace(event)` appends to /data/trace.jsonl.
        """

    def trace(self, event: dict[str, Any]) -> None:
        """Append one JSON event to the optional replay trace."""
        with TRACE_PATH.open("a") as f:
            f.write(json.dumps(event) + "\n")

    def run(self) -> None:
        """Entry point for the referee image. Parses env, runs the game, writes result.json."""
        ctx = RefereeContext.from_env()
        players = [PlayerClient(url) for url in ctx.player_urls]
        for p in players:
            p.wait_until_ready(self.readiness_timeout_s)

        result = self.play_game(ctx, players)  # unhandled exceptions propagate -> no result.json

        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(asdict(result)))

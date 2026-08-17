# VENDORED from macrocosm-os/apex-competitions-builder @ d063d9028dbec4bb15182794496f4aa2aac19d49
# (src/apex_sdk/gym_v1/), import root rewritten apex_sdk.gym_v1 -> gym_v1.
# Do not hand-edit; re-vendor from the toolkit to update.

"""Referee-side HTTP client for talking to a gym_v1 player sandbox.

The referee constructs one PlayerClient per URL in PLAYER_URLS and drives the match
through it. Transport only — observation/action shapes are opaque and competition-defined.

Wire protocol (gym_v1):
    GET  /health                                  -> {"ready": bool}
    POST /reset  {match_id, player_index, seed, config} -> 204
    POST /act    {observation, deadline_ms}       -> {"action": ...}

Dependency-light on purpose (stdlib urllib) so the referee base image stays small.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class PlayerError(RuntimeError):
    """A player was unreachable, errored, or timed out.

    The referee decides what this means for the game (forfeit / retry / draw).
    The platform does not intervene in player-level failures.
    """


class PlayerClient:
    def __init__(self, base_url: str, connect_timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.connect_timeout_s = connect_timeout_s

    def _post(self, path: str, body: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any]]:
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
                raw = resp.read()
        except urllib.error.URLError as e:
            raise PlayerError(f"POST {path} to {self.base_url} failed: {e}") from e
        except TimeoutError as e:
            raise PlayerError(f"POST {path} to {self.base_url} timed out after {timeout_s}s") from e
        payload = json.loads(raw) if raw else {}
        return status, payload

    def health(self) -> bool:
        """Return True if the player reports ready. Never raises (returns False on error)."""
        try:
            req = urllib.request.Request(f"{self.base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=self.connect_timeout_s) as resp:
                if resp.status != 200:
                    return False
                payload = json.loads(resp.read() or b"{}")
                return bool(payload.get("ready", False))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return False

    def wait_until_ready(self, timeout_s: float, poll_interval_s: float = 0.5) -> None:
        """Block until the player is ready or raise PlayerError on timeout."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.health():
                return
            time.sleep(poll_interval_s)
        raise PlayerError(f"player {self.base_url} not ready within {timeout_s}s")

    def reset(
        self, match_id: str, player_index: int, seed: int, config: dict[str, Any], timeout_s: float = 30.0
    ) -> None:
        """Start a match for this player. Called once per match per player."""
        status, _ = self._post(
            "/reset",
            {"match_id": match_id, "player_index": player_index, "seed": seed, "config": config},
            timeout_s,
        )
        if status != 204:
            raise PlayerError(f"/reset to {self.base_url} returned {status}, expected 204")

    def act(self, observation: Any, deadline_ms: int, timeout_s: float | None = None) -> Any:
        """Request an action for the given observation. Returns the opaque action.

        deadline_ms is the per-turn budget the player must respect; we also enforce it
        as the HTTP timeout (plus a small grace) so a hung player cannot stall the match.
        """
        if timeout_s is None:
            timeout_s = deadline_ms / 1000.0 + 1.0
        status, payload = self._post("/act", {"observation": observation, "deadline_ms": deadline_ms}, timeout_s)
        if status != 200:
            raise PlayerError(f"/act to {self.base_url} returned {status}, expected 200")
        if "action" not in payload:
            raise PlayerError(f"/act response from {self.base_url} missing 'action' key")
        return payload["action"]

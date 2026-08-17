# VENDORED from macrocosm-os/apex-competitions-builder @ d063d9028dbec4bb15182794496f4aa2aac19d49
# (src/apex_sdk/gym_v1/), import root rewritten apex_sdk.gym_v1 -> gym_v1.
# Do not hand-edit; re-vendor from the toolkit to update.

"""Player-side server base for gym_v1.

A competition's player image implements a `Player` (loading the miner's submitted
artifact from `target_path`) and calls `serve(...)`. This exposes the gym_v1 HTTP API
the platform's readiness probe and the referee drive:

    GET  /health  -> {"ready": bool}
    POST /reset   {match_id, player_index, seed, config} -> 204
    POST /act     {observation, deadline_ms} -> {"action": ...}

Stdlib http.server only, so the player base image needs no web framework.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class Player(ABC):
    """Implement this in the player image, wrapping the miner's submission."""

    def is_ready(self) -> bool:
        """Override if the submission needs warm-up (e.g. loading a model). Default: ready."""
        return True

    @abstractmethod
    def reset(self, match_id: str, player_index: int, seed: int, config: dict[str, Any]) -> None:
        """Begin a new match. Store whatever per-match state you need."""

    @abstractmethod
    def act(self, observation: Any, deadline_ms: int) -> Any:
        """Return an action for the observation within deadline_ms. Shapes are competition-defined."""


def _make_handler(player: Player, readiness_path: str):
    class Handler(BaseHTTPRequestHandler):
        # Silence default stderr logging; competitions can add their own.
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            return json.loads(raw) if raw else {}

        def do_GET(self) -> None:  # noqa: N802
            if self.path == readiness_path:
                try:
                    ready = bool(player.is_ready())
                except Exception:
                    ready = False
                self._send_json(200, {"ready": ready})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._read_json()
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid json"})
                return

            if self.path == "/reset":
                player.reset(
                    match_id=body["match_id"],
                    player_index=body["player_index"],
                    seed=body["seed"],
                    config=body.get("config", {}),
                )
                self.send_response(204)
                self.end_headers()
            elif self.path == "/act":
                action = player.act(observation=body.get("observation"), deadline_ms=body["deadline_ms"])
                self._send_json(200, {"action": action})
            else:
                self._send_json(404, {"error": "not found"})

    return Handler


def serve(player: Player, port: int, readiness_path: str = "/health", host: str = "0.0.0.0") -> None:
    """Run the gym_v1 HTTP server. Blocks forever (this is the sandbox's main process)."""
    server = ThreadingHTTPServer((host, port), _make_handler(player, readiness_path))
    server.serve_forever()

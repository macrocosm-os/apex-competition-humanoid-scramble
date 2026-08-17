# VENDORED from macrocosm-os/apex-competitions-builder @ d063d9028dbec4bb15182794496f4aa2aac19d49
# (src/apex_sdk/gym_v1/), import root rewritten apex_sdk.gym_v1 -> gym_v1.
# Do not hand-edit; re-vendor from the toolkit to update.

"""The gym_v1 duel protocol: player server base, referee harness, and referee-side client."""

from gym_v1.client import PlayerClient, PlayerError
from gym_v1.player import Player, serve
from gym_v1.referee import GameResult, Referee, RefereeContext

__all__ = [
    "Player",
    "serve",
    "PlayerClient",
    "PlayerError",
    "Referee",
    "GameResult",
    "RefereeContext",
]

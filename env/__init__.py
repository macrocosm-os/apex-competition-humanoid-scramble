"""Humanoid Parkour environment: course, physics, scoring.

Shared by the referee image and the local tools so the numbers can never diverge.
"""

from .course import N_BOXES, ROOM_LENGTH, TRACK_HALF_W
from .scoring import instance_score
from .sim import (ACT_DIM, OBS_DIM, STATE_DIM, InstanceParams, InvalidAction, ParkourSim,
                  instance_spec)

__all__ = ["ACT_DIM", "ROOM_LENGTH", "OBS_DIM", "STATE_DIM", "InstanceParams", "InvalidAction",
           "ParkourSim", "N_BOXES", "TRACK_HALF_W", "instance_score", "instance_spec"]

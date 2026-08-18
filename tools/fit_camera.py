"""Analytic (non-iterative) camera fit for course preview renders.

Computes the exact lookat + distance needed so the room's full bounding box (including the
tallest possible climb stack and the robot's standing height) fits tightly inside the frame at a
given azimuth/elevation, in one closed-form pass -- no iterative render-measure-adjust loop
(which is fragile and can diverge, as an earlier attempt at this exact problem did).

Usage: python -m tools.fit_camera
"""
from __future__ import annotations

import math

import numpy as np

from env.course import ROOM_LENGTH, TRACK_HALF_W


def camera_basis(az_deg: float, el_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right, up, forward unit vectors for MuJoCo's azimuth/elevation camera convention."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    cx, cy, cz = math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)
    forward = -np.array([cx, cy, cz])
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return right, up, forward


def fit_room_camera(az_deg: float = 132.0, el_deg: float = -20.0, fovy_deg: float = 45.0,
                     aspect: float = 16 / 9, margin: float = 1.03,
                     room_height: float = 2.6) -> tuple[np.ndarray, float]:
    """Return (lookat, distance) that fits the room's bounding box tightly at this az/el.

    Closed-form: project all 8 room-bbox corners onto the camera's right/up axes, take the
    resulting 2D bounding box, and solve directly for the distance (from the vertical/horizontal
    FOV half-angles) and the lookat point (room's true centroid -- NOT re-derived from pixels,
    since the room bbox is symmetric about its own centroid by construction, so the projected
    bbox is automatically centered on the projected centroid; no recentring loop needed).
    """
    right, up, forward = camera_basis(az_deg, el_deg)
    room_center = np.array([ROOM_LENGTH / 2, 0.0, room_height / 2])
    half_extents = np.array([ROOM_LENGTH / 2, TRACK_HALF_W, room_height / 2])

    corners = [room_center + np.array([sx, sy, sz]) * half_extents
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    rel = [c - room_center for c in corners]
    max_horiz = max(abs(np.dot(v, right)) for v in rel)
    max_vert = max(abs(np.dot(v, up)) for v in rel)

    fovy = math.radians(fovy_deg)
    half_fovy = fovy / 2
    half_fovx = math.atan(math.tan(half_fovy) * aspect)
    dist_v = max_vert / math.tan(half_fovy)
    dist_h = max_horiz / math.tan(half_fovx)
    distance = max(dist_v, dist_h) * margin

    return room_center, distance


if __name__ == "__main__":
    lookat, distance = fit_room_camera()
    print(f"lookat = {lookat}")
    print(f"distance = {distance:.3f}")

# humanoid_parkour REFEREE image (the scorer sandbox: physics + ground truth + scoring).
#
# VENDORED gym_v1 pattern: build FROM python:3.12-slim and COPY the vendored gym_v1/ in — we do
# NOT `FROM apex-referee-base` (not published to any registry; would fail in release CI).
#
# Owns MuJoCo, the course, the G1 model, and every termination + scoring gate. Runs by the platform
# convention at /app/referee.py; the platform injects MATCH_ID, SEED, CONFIG_JSON, PLAYER_URLS,
# NUM_PLAYERS and reads /data/result.json.
#
# Build (context = THIS repo root):
#   docker build -f referee/Dockerfile -t humanoid-parkour-referee .
FROM python:3.12-slim

# Match how the platform runs sandboxes: non-root uid 1000, /app + /data owned by it. /data is
# where the referee writes result.json.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app /data && chown -R 1000:1000 /app /data
WORKDIR /app

# Exact pins: physics must be reproducible across evaluations, so the same (policy, seed) pair
# always scores the same. Score drift is indistinguishable from cheating.
RUN pip install --no-cache-dir numpy==2.3.4 mujoco==3.11.0

COPY referee/gym_v1/ /app/gym_v1/
# env/ carries the vendored Unitree G1 collision meshes (env/assets/, ~24 MB) as well as the
# course and scoring code. They are the robot: without them the model will not compile.
COPY env/ /app/env/
COPY referee/referee.py /app/referee.py

USER 1000

# spec.referee runs /app/referee.py by convention.
CMD ["python", "/app/referee.py"]

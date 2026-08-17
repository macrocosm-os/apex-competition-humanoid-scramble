# humanoid_parkour PLAYER image (the miner submission server).
#
# VENDORED gym_v1 pattern: build FROM python:3.12-slim and COPY the vendored gym_v1/ in — we do
# NOT `FROM apex-player-base` (that base is not published to any registry, so it only resolves on
# a machine that built it locally and fails in release CI).
#
# The submission is NOT baked in: the platform writes the miner's ONNX policy to
# /app/submission.onnx (spec `submission.target_path`) at run time.
#
# This image is safe to publish: it contains no course generator, no seeds, and no scoring logic
# (all of that lives in the referee image).
#
# Build (context = THIS repo root):
#   docker build -f player/Dockerfile -t humanoid-parkour-player .
FROM python:3.12-slim

# Match how the platform runs sandboxes: non-root uid 1000, /app + /data owned by it.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app /data && chown -R 1000:1000 /app /data
WORKDIR /app

# Exact pins: the same policy must produce the same actions on every evaluation.
RUN pip install --no-cache-dir numpy==2.3.4 onnxruntime==1.28.0

COPY player/gym_v1/ /app/gym_v1/
COPY player/launch.py /app/launch.py

USER 1000

# spec.entrypoints.evaluate.command is ["python", "/app/launch.py", "--port", "8000"].
CMD ["python", "/app/launch.py", "--port", "8000"]

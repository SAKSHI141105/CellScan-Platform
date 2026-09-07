# Single-service deploy: this image builds the React frontend, then serves it
# and the FastAPI backend from one Python process (see src/api/main.py's
# StaticFiles mount) — one origin, no CORS config needed in production.

# ---- stage 1: build the React app ----
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- stage 2: Python backend + the built frontend ----
FROM python:3.11-slim AS backend

# libgl1/libglib2.0-0: opencv-python-headless still needs these at import time
# even in the "headless" build. curl: fetches the two large model checkpoints
# from GitHub Release assets below (they're too big for a normal git push).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# torch/torchvision from the CPU-only wheel index — the default PyPI wheel
# bundles CUDA and is multiple GB larger than this app (a CPU inference
# service) will ever use
RUN grep -vE '^(torch|torchvision)' requirements.txt > requirements.nocuda.txt \
    && pip install --no-cache-dir -r requirements.nocuda.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/
COPY run.py .

# Small real trained checkpoints come straight from the build context — they
# ARE git-tracked (see .gitignore's exception for data/models/tabular/* and
# custom_cnn_final.h5), so a remote build (Render clones this repo, it
# doesn't see the local filesystem) has them too.
COPY data/models/tabular/ data/models/tabular/
COPY data/models/image/custom_cnn_final.h5 data/models/image/

# The two large checkpoints (transfer_resnet50_final.h5 ~169MB,
# lesion_guided_resnet50.pt ~130MB) exceed GitHub's 100MB per-file push
# limit, so they can't live in the repo the normal way — hosted as GitHub
# Release assets instead (2GB/file limit there) and pulled in at build time.
# Hardcoded rather than routed through Docker build args: Render's
# render.yaml Blueprint spec has no field for passing --build-arg values
# (confirmed against Render's own docs), and these URLs aren't secret —
# they're public release-asset download links — so an ARG indirection
# would add a configuration step with no actual benefit.
RUN mkdir -p data/models/image data/models/mammography \
    && curl -fL "https://github.com/SAKSHI141105/CellScan-Platform/releases/download/v1.0-models/transfer_resnet50_final.h5" \
         -o data/models/image/transfer_resnet50_final.h5 \
    && curl -fL "https://github.com/SAKSHI141105/CellScan-Platform/releases/download/v1.0-models/lesion_guided_resnet50.pt" \
         -o data/models/mammography/lesion_guided_resnet50.pt

# Demo weights only where a real checkpoint is genuinely absent — a no-op
# for both pipelines above (the curl steps hard-fail the build on error, so
# by the time we get here both real files exist), kept as a safety net for
# whichever pipeline doesn't have one if this Dockerfile is ever reused
# against a repo/release that doesn't have these assets.
RUN python -m scripts.generate_demo_weights --target all

COPY --from=frontend-build /app/frontend/dist frontend/dist

EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

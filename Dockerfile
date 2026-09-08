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
# even in the "headless" build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
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
# lesion_guided_resnet50.pt ~130MB — both available as GitHub Release
# assets on this repo, see v1.0-models) are deliberately NOT pulled in for
# this free-tier deploy target. TensorFlow + PyTorch loaded together
# already sit close to Render's free-tier 512MB ceiling on their own
# (confirmed by an actual OOM crash-loop in production logs); adding
# either large checkpoint tips it over. custom_cnn_final.h5 above is real
# and git-tracked, so histopathology still serves genuine (not demo)
# predictions — from the smaller custom CNN instead of the stronger
# transfer-learning ResNet50 — and mammography falls back to demo weights
# below. If you deploy this to a host with more RAM (2GB+), re-add the
# curl downloads from the two URLs above to get the full-strength models.
RUN python -m scripts.generate_demo_weights --target all

COPY --from=frontend-build /app/frontend/dist frontend/dist

EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

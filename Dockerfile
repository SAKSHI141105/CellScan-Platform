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
# Set these as build args (Render: service Settings -> Environment ->
# "Build" section, or --build-arg locally) to the release asset URLs.
ARG TRANSFER_RESNET50_URL
ARG MAMMOGRAPHY_CHECKPOINT_URL
RUN mkdir -p data/models/image data/models/mammography \
    && if [ -n "$TRANSFER_RESNET50_URL" ]; then \
         curl -fL "$TRANSFER_RESNET50_URL" -o data/models/image/transfer_resnet50_final.h5; \
       else echo "TRANSFER_RESNET50_URL not set — histopathology will run in demo mode" >&2; fi \
    && if [ -n "$MAMMOGRAPHY_CHECKPOINT_URL" ]; then \
         curl -fL "$MAMMOGRAPHY_CHECKPOINT_URL" -o data/models/mammography/lesion_guided_resnet50.pt; \
       else echo "MAMMOGRAPHY_CHECKPOINT_URL not set — mammography will run in demo mode" >&2; fi

# Demo weights only where a real checkpoint is genuinely absent (e.g. the
# build args above weren't set) — a no-op for any pipeline that already got
# a real file above, since load_image_model()/load_mammography_model() only
# touch these when the real path doesn't exist.
RUN python -m scripts.generate_demo_weights --target all

COPY --from=frontend-build /app/frontend/dist frontend/dist

EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

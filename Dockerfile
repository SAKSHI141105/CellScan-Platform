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
# even in the "headless" build
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

# Pre-build the demo checkpoints at image build time, not on first request —
# generate_demo_weights.py downloads ImageNet-pretrained backbone weights
# from the internet, which would otherwise repeat on every cold start on a
# host that spins the container down when idle (e.g. Render's free tier)
RUN python -m scripts.generate_demo_weights --target all

COPY --from=frontend-build /app/frontend/dist frontend/dist

EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

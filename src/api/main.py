"""CellScan API — wraps the tabular/image ML pipelines for the React frontend.

    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routers import clusters, image, mammography, reports, tabular
from src.services import image_service, mammography_service
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _warm_image_models():
    # TensorFlow's import alone takes 15-20s on a cold process — lazily
    # eating that cost on someone's *first* prediction request looks
    # indistinguishable from a hung/broken server from the browser side.
    # Loading both model services here means that cost lands during server
    # boot (visible in the startup logs) instead of during a user's first
    # upload.
    try:
        image_service.load_image_model()
    except Exception:
        logger.exception("Histopathology model warmup failed — first real request will retry and surface the error")
    try:
        mammography_service.load_mammography_model()
    except Exception:
        logger.exception("Mammography model warmup failed — first real request will retry and surface the error")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # daemon threads so a slow/failed warmup never blocks shutdown
    threading.Thread(target=clusters.warm_projection_cache, daemon=True).start()
    threading.Thread(target=_warm_image_models, daemon=True).start()
    yield


app = FastAPI(title="CellScan API", version="1.0.0", lifespan=lifespan)

_extra_origins = [o.strip() for o in os.environ.get("CELLSCAN_EXTRA_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    # Vite's default dev port is 5173; 4173 is its preview-build port.
    # CELLSCAN_EXTRA_ORIGINS lets a split frontend/backend deploy (frontend on
    # its own host, e.g. Vercel) add its origin without a code change — the
    # default same-origin deploy (this process serving frontend/dist too)
    # doesn't need it at all, since same-origin requests aren't subject to CORS.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", *_extra_origins],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tabular.router)
app.include_router(image.router)
app.include_router(mammography.router)
app.include_router(reports.router)
app.include_router(clusters.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve the built React app (frontend/dist) from this same process in
# production, so the deployed app is one service/one origin — matching the
# frontend's own assumption (frontend/src/lib/api.ts calls relative "/api/..."
# paths specifically so a same-origin deploy needs no base-URL config). Only
# activates if the build actually exists; local dev keeps using `npm run dev`
# on :5173 with its own proxy, so this is a no-op there.
_frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/assets", StaticFiles(directory=_frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # any real static file at the root (favicon, manifest, etc.) wins;
        # everything else falls through to index.html so React Router's
        # client-side routes (e.g. /upload-image on a hard refresh) resolve
        # instead of 404ing against a path the server itself doesn't know
        candidate = _frontend_dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_frontend_dist / "index.html")

"""Authenticated CPU-stub and optional, explicitly approved SAM 3D service.

The real backend lazily loads official inference and exports output['glb'].
It never substitutes a synthetic mesh for an unavailable real model.
"""
from __future__ import annotations

import argparse
import hmac
import os
from typing import Literal

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
import numpy as np
from starlette.concurrency import run_in_threadpool
from splatsam.recon.gpu_client import ReconstructionError
from splatsam.recon.sam3d_backend import Sam3dBackend

from splatsam.recon.gpu_client import StubReconstructor


Backend = Literal["stub", "sam3d"]


def _read_png(payload: bytes, name: str) -> np.ndarray:
    raster = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raster is None:
        raise HTTPException(status_code=422, detail=f"{name} must be a decodable PNG")
    return raster


def _decode_inputs(image_data: bytes, mask_data: bytes) -> tuple[np.ndarray, np.ndarray]:
    image = _read_png(image_data, "image")
    mask = _read_png(mask_data, "mask")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise HTTPException(status_code=422, detail="image must be an RGB PNG")
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2 or mask.shape != image.shape[:2] or not np.any(mask):
        raise HTTPException(status_code=422, detail="mask must be nonempty and match image dimensions")
    return np.ascontiguousarray(image), np.ascontiguousarray(mask > 0)


def _require_bearer(request: Request) -> None:
    expected = os.environ.get("SPLATSAM_RECON_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="reconstruction token is not configured")
    header = request.headers.get("Authorization", "")
    prefix = "Bearer "
    if not header.startswith(prefix) or not hmac.compare_digest(header[len(prefix) :], expected):
        raise HTTPException(status_code=401, detail="bearer token required")


def create_app(backend: Backend = "stub") -> FastAPI:
    if backend not in {"stub", "sam3d"}:
        raise ValueError("backend must be 'stub' or 'sam3d'")
    app = FastAPI(title="Splatsam reconstruction service", docs_url=None, redoc_url=None)
    stub = StubReconstructor()
    sam3d = Sam3dBackend()

    @app.get("/health")
    def health() -> dict[str, object]:
        if backend == "sam3d":
            available, reason = sam3d.availability()
            return {"ok": available, "backend": backend, "available": available, "fake": False, "reason": reason, "model_loaded": sam3d._inference is not None}
        return {"ok": True, "backend": backend, "available": True, "fake": True}

    @app.post("/reconstruct")
    async def reconstruct(
        request: Request,
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        seed: int = Form(...),
    ) -> Response:
        _require_bearer(request)
        if seed < 0:
            raise HTTPException(status_code=422, detail="seed must be nonnegative")
        image_data, mask_data = await image.read(32 * 1024 * 1024 + 1), await mask.read(32 * 1024 * 1024 + 1)
        if max(len(image_data), len(mask_data)) > 32 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="image and mask must each be at most 32 MiB")
        rgb, binary = _decode_inputs(image_data, mask_data)
        try:
            payload = await run_in_threadpool(sam3d.reconstruct, rgb, binary, seed) if backend == "sam3d" else stub.reconstruct(rgb, binary)
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Reconstruction unavailable ({type(error).__name__}); use primitive fallback") from error
        return Response(
            content=payload,
            media_type="model/gltf-binary",
            headers={"X-Splatsam-Reconstruction-Backend": backend, "X-Splatsam-Reconstruction-Fake": "true" if backend == "stub" else "false"},
        )

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Splatsam optional reconstruction service")
    parser.add_argument("--backend", choices=("stub", "sam3d"), default="stub")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.backend), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

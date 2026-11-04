"""Optional remote mesh transport with conservative primitive fallback.

Remote meshes are transport artifacts only.  The caller must explicitly choose a
metric-fitted mesh before it can affect simulation geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from typing import Protocol
from diskcache import Cache

import cv2
import httpx
import numpy as np
import trimesh

from splatsam.llm.schemas import ObjectSpec


class ReconstructionError(RuntimeError):
    """A remote reconstruction response was unavailable or unsafe to consume."""


class Reconstructor(Protocol):
    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> bytes: ...


@dataclass(frozen=True)
class ReconstructionProvenance:
    source: str
    backend: str
    detail: str = ""
    simulation_eligible: bool = False
    scale_source: str | None = None


def _as_rgb(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] not in (3, 4):
        raise ValueError("image must be an HxWx3/4 RGB raster")
    if value.dtype != np.uint8:
        raise ValueError("image must use uint8 samples")
    return np.ascontiguousarray(value[..., :3])


def _as_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim == 3 and value.shape[2] == 1:
        value = value[..., 0]
    if value.ndim != 2 or value.shape != image_shape:
        raise ValueError("mask must be a 2-D raster aligned to image")
    result = np.ascontiguousarray((value > 0).astype(np.uint8))
    if not result.any():
        raise ValueError("mask must not be empty")
    return result


def _glb(mesh: trimesh.Trimesh) -> bytes:
    mesh.remove_unreferenced_vertices()
    blob = trimesh.exchange.gltf.export_glb(mesh)
    validate_glb(blob)
    return blob


def validate_glb(payload: bytes) -> None:
    """Reject non-GLB, empty, or numerically unsafe triangle payloads."""
    if not isinstance(payload, bytes) or len(payload) < 20 or payload[:4] != b"glTF":
        raise ReconstructionError("response is not a GLB")
    try:
        loaded = trimesh.load(io.BytesIO(payload), file_type="glb", force="scene")
    except Exception as error:  # trimesh raises several parser-specific exception types.
        raise ReconstructionError("response GLB cannot be parsed") from error
    meshes = list(loaded.geometry.values()) if isinstance(loaded, trimesh.Scene) else [loaded]
    if not meshes:
        raise ReconstructionError("response GLB has no mesh geometry")
    valid = False
    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ReconstructionError("response GLB has invalid mesh arrays")
        if len(vertices) == 0 or len(faces) == 0 or not np.isfinite(vertices).all() or not np.isfinite(faces).all():
            raise ReconstructionError("response GLB has empty or non-finite mesh")
        if not np.issubdtype(faces.dtype, np.integer):
            raise ReconstructionError("response GLB face indices are not integers")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ReconstructionError("response GLB has invalid face indices")
        valid = True
    if not valid:
        raise ReconstructionError("response GLB has no triangle mesh")


def _box_from_mask(mask: np.ndarray, *, depth_ratio: float = 0.2) -> trimesh.Trimesh:
    ys, xs = np.nonzero(mask)
    width = max(1, int(xs.max()) - int(xs.min()) + 1)
    height = max(1, int(ys.max()) - int(ys.min()) + 1)
    largest = float(max(width, height))
    # This deliberately has no physical units. It is a transport-compatible
    # assumption, not geometry eligible for a MuJoCo scene without metric fit.
    extents = (width / largest, height / largest, max(0.01, min(width, height) / largest * depth_ratio))
    return trimesh.creation.box(extents=extents)


class PrimitiveReconstructor:
    """Export a native primitive as GLB, never selecting it for simulation."""

    def __init__(self, object_spec: ObjectSpec | None = None) -> None:
        self.object_spec = object_spec
        self.last_provenance = ReconstructionProvenance(
            source="primitive", backend="primitive", simulation_eligible=False
        )

    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> bytes:
        rgb = _as_rgb(image)
        binary = _as_mask(mask, rgb.shape[:2])
        if self.object_spec is None:
            mesh = _box_from_mask(binary)
            self.last_provenance = ReconstructionProvenance(
                source="primitive",
                backend="primitive",
                detail="mask_bbox_normalized",
                simulation_eligible=False,
                scale_source="normalized_mask_bbox_estimate",
            )
            return _glb(mesh)
        spec = self.object_spec
        size = np.asarray(spec.size, dtype=float)
        if not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("ObjectSpec size must be positive and finite")
        if spec.shape == "box" or spec.shape == "mesh":
            mesh = trimesh.creation.box(extents=size)
        elif spec.shape == "sphere":
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]) / 2.0)
        elif spec.shape == "cylinder":
            mesh = trimesh.creation.cylinder(radius=float(size[0]) / 2.0, height=float(size[2]), sections=32)
        elif spec.shape == "horizontal_cylinder":
            mesh = trimesh.creation.cylinder(radius=float(size[1]) / 2.0, height=float(size[0]), sections=32)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [0.0, 1.0, 0.0]))
        else:  # Pydantic protects this, but keep this adapter safe for duck-typed callers.
            raise ValueError(f"unsupported primitive shape: {spec.shape}")
        self.last_provenance = ReconstructionProvenance(
            source="primitive", backend="primitive", detail="object_spec", simulation_eligible=False
        )
        return _glb(mesh)


class StubReconstructor:
    """Deterministic CPU test backend; its fake mesh is always conspicuously labeled."""

    def __init__(self) -> None:
        self.last_provenance = ReconstructionProvenance(
            source="stub", backend="stub", detail="fake_test_mesh", simulation_eligible=False
        )

    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> bytes:
        rgb = _as_rgb(image)
        binary = _as_mask(mask, rgb.shape[:2])
        mesh = _box_from_mask(binary, depth_ratio=0.1)
        self.last_provenance = ReconstructionProvenance(
            source="stub", backend="stub", detail="fake_test_mesh", simulation_eligible=False
        )
        return _glb(mesh)


class RemoteSam3dReconstructor:
    """One-shot HTTP transport with a content-addressed cache and safe fallback."""

    def __init__(
        self,
        base_url: str,
        token: str,
        cache_dir: Path,
        *,
        model: str = "sam3d",
        backend: str = "sam3d",
        timeout_s: float = 20.0,
        allow_stub: bool = False,
        fallback: PrimitiveReconstructor | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.cache_dir = Path(cache_dir)
        self.cache = Cache(str(self.cache_dir / "inference"))
        self.model = model
        self.backend = backend
        self.timeout_s = float(timeout_s)
        self.allow_stub = allow_stub
        self.fallback = fallback or PrimitiveReconstructor()
        self.last_provenance = ReconstructionProvenance("unattempted", backend)

    def _key(self, image: np.ndarray, mask: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(b"splatsam-remote-glb-v2\0")
        for value in (self.model, self.backend, str(self.allow_stub), str(image.shape), str(image.dtype), str(mask.shape), str(mask.dtype)):
            digest.update(value.encode("utf-8")); digest.update(b"\0")
        digest.update(image.tobytes()); digest.update(mask.tobytes())
        return digest.hexdigest()


    def _fallback(self, image: np.ndarray, mask: np.ndarray, reason: str) -> bytes:
        result = self.fallback.reconstruct(image, mask)
        self.last_provenance = ReconstructionProvenance(
            source="fallback", backend="primitive", detail=reason, simulation_eligible=False,
            scale_source=self.fallback.last_provenance.scale_source,
        )
        return result

    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> bytes:
        rgb = _as_rgb(image)
        binary = _as_mask(mask, rgb.shape[:2])
        key = self._key(rgb, binary)
        cached = self.cache.get(key)
        if isinstance(cached, dict):
            try:
                payload = cached["payload"]
                if cached.get("fake") and not self.allow_stub:
                    raise ReconstructionError("cached stub is forbidden by this client policy")
                validate_glb(payload)
            except (KeyError, TypeError, ReconstructionError):
                self.cache.delete(key)
            else:
                self.last_provenance = ReconstructionProvenance("cache", cached["backend"], simulation_eligible=False)
                return payload
        try:
            ok_image, encoded_image = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            ok_mask, encoded_mask = cv2.imencode(".png", binary * 255)
            if not ok_image or not ok_mask:
                raise ReconstructionError("could not encode request PNGs")
            timeout = httpx.Timeout(self.timeout_s, connect=min(self.timeout_s, 5.0))
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{self.base_url}/reconstruct",
                    headers={"Authorization": f"Bearer {self.token}"},
                    files={
                        "image": ("image.png", encoded_image.tobytes(), "image/png"),
                        "mask": ("mask.png", encoded_mask.tobytes(), "image/png"),
                    },
                    data={"seed": "42"},
                )
            if response.status_code != 200:
                raise ReconstructionError(f"remote status {response.status_code}")
            remote_backend = response.headers.get("X-Splatsam-Reconstruction-Backend", "")
            fake = remote_backend == "stub" or response.headers.get("X-Splatsam-Reconstruction-Fake", "").lower() == "true"
            if fake and not self.allow_stub:
                raise ReconstructionError("stub backend is not permitted for remote reconstruction")
            if not remote_backend or (remote_backend != self.backend and not (remote_backend == "stub" and self.allow_stub)):
                raise ReconstructionError(f"unexpected remote backend {remote_backend!r}")
            payload = response.content
            validate_glb(payload)
        except (httpx.HTTPError, OSError, ReconstructionError, cv2.error) as error:
            return self._fallback(rgb, binary, str(error))
        self.cache.set(key, {"payload": payload, "backend": remote_backend, "fake": fake})
        self.last_provenance = ReconstructionProvenance("remote", remote_backend or self.backend, simulation_eligible=False)
        return payload


__all__ = [
    "PrimitiveReconstructor", "ReconstructionError", "ReconstructionProvenance", "Reconstructor",
    "RemoteSam3dReconstructor", "StubReconstructor", "validate_glb",
]

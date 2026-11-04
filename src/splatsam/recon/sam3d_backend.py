"""Lazy official SAM 3D adapter; importing this module never initializes a GPU.

Verified upstream contract: notebook/inference.py Inference(config_file,
compile=False)(image, mask, seed=...), with triangle mesh in output['glb'].
See sam3d_objects/pipeline/inference_pipeline.py:postprocess_slat_output.
Weights/license/provider approval remain operator-owned Phase B prerequisites.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path

import numpy as np

from splatsam.recon.gpu_client import ReconstructionError, validate_glb


class Sam3dBackend:
    def __init__(self):
        self._inference = None
        self._lock = threading.Lock()

    def availability(self) -> tuple[bool, str]:
        if os.environ.get("SPLATSAM_PHASE") != "B" or os.environ.get("SPLATSAM_GPU_APPROVED") != "1":
            return False, "Phase B and explicit GPU spend approval required"
        root = os.environ.get("SPLATSAM_SAM3D_ROOT")
        config = os.environ.get("SPLATSAM_SAM3D_CONFIG")
        if not root or not config:
            return False, "Approved upstream checkout and downloaded pipeline configuration required"
        if not (Path(root) / "notebook/inference.py").is_file() or not Path(config).is_file():
            return False, "Upstream inference code or downloaded pipeline configuration missing"
        return True, "configured; model loads lazily on first authorized reconstruction"

    def reconstruct(self, image: np.ndarray, mask: np.ndarray, seed: int) -> bytes:
        available, reason = self.availability()
        if not available:
            raise ReconstructionError(reason)
        with self._lock:
            if self._inference is None:
                path = Path(os.environ["SPLATSAM_SAM3D_ROOT"]) / "notebook/inference.py"
                spec = importlib.util.spec_from_file_location("splatsam_official_sam3d", path)
                if spec is None or spec.loader is None:
                    raise ReconstructionError("Upstream inference loader unavailable")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self._inference = module.Inference(os.environ["SPLATSAM_SAM3D_CONFIG"], compile=False)
            output = self._inference(image, mask, seed=seed)
            mesh = output.get("glb")
            if mesh is None or not callable(getattr(mesh, "export", None)):
                raise ReconstructionError("SAM 3D returned no triangle mesh; Gaussian PLY is not a GLB")
            payload = mesh.export(file_type="glb")
            if not isinstance(payload, bytes):
                raise ReconstructionError("SAM 3D GLB export was not binary")
            validate_glb(payload)
            return payload

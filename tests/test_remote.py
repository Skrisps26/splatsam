from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from splatsam.recon import gpu_client
from splatsam.recon.gpu_client import RemoteSam3dReconstructor, StubReconstructor, validate_glb
from splatsam.recon.server import create_app


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[..., 0] = 128
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[5:18, 8:25] = 1
    return image, mask


def test_stub_server_is_authenticated_and_returns_valid_labeled_glb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPLATSAM_RECON_TOKEN", "secret")
    image, mask = _inputs()
    client = TestClient(create_app("stub"))
    response = client.post(
        "/reconstruct",
        files={
            "image": ("image.png", gpu_client.cv2.imencode(".png", image[..., ::-1])[1].tobytes(), "image/png"),
            "mask": ("mask.png", gpu_client.cv2.imencode(".png", mask * 255)[1].tobytes(), "image/png"),
        },
        data={"seed": "42"},
    )
    assert response.status_code == 401
    response = client.post(
        "/reconstruct",
        headers={"Authorization": "Bearer secret"},
        files={
            "image": ("image.png", gpu_client.cv2.imencode(".png", image[..., ::-1])[1].tobytes(), "image/png"),
            "mask": ("mask.png", gpu_client.cv2.imencode(".png", mask * 255)[1].tobytes(), "image/png"),
        },
        data={"seed": "42"},
    )
    assert response.status_code == 200
    assert response.headers["X-Splatsam-Reconstruction-Backend"] == "stub"
    validate_glb(response.content)


class _Response:
    def __init__(self, status_code: int, content: bytes, backend: str = "sam3d") -> None:
        self.status_code = status_code
        self.content = content
        self.headers = {"X-Splatsam-Reconstruction-Backend": backend}


class _Client:
    response: _Response
    calls = 0

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def post(self, *args: object, **kwargs: object) -> _Response:
        type(self).calls += 1
        return type(self).response


def test_remote_success_is_content_cached_without_any_second_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image, mask = _inputs()
    _Client.calls = 0
    _Client.response = _Response(200, StubReconstructor().reconstruct(image, mask))
    monkeypatch.setattr(gpu_client.httpx, "Client", _Client)
    remote = RemoteSam3dReconstructor("http://gone", "token", tmp_path, timeout_s=1)
    first = remote.reconstruct(image, mask)
    assert remote.last_provenance.source == "remote"
    assert _Client.calls == 1

    class _NoNetwork:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("cache hit must not initialize HTTP client")

    monkeypatch.setattr(gpu_client.httpx, "Client", _NoNetwork)
    assert remote.reconstruct(image, mask) == first
    assert remote.last_provenance.source == "cache"


@pytest.mark.parametrize(
    ("response", "expected"),
    [(_Response(401, b""), "remote status 401"), (_Response(200, b"not-a-glb"), "response is not a GLB")],
)
def test_auth_and_invalid_mesh_fall_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: _Response, expected: str) -> None:
    image, mask = _inputs()
    _Client.calls = 0
    _Client.response = response
    monkeypatch.setattr(gpu_client.httpx, "Client", _Client)
    remote = RemoteSam3dReconstructor("http://remote", "token", tmp_path)
    payload = remote.reconstruct(image, mask)
    validate_glb(payload)
    assert _Client.calls == 1
    assert remote.last_provenance.source == "fallback"
    assert expected in remote.last_provenance.detail
    remote.reconstruct(image, mask)
    assert _Client.calls == 2  # Failed responses must not poison the success cache.


def test_network_error_falls_back_without_a_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image, mask = _inputs()

    class _FailingClient(_Client):
        calls = 0

        def post(self, *args: object, **kwargs: object) -> _Response:
            type(self).calls += 1
            raise gpu_client.httpx.ConnectError("offline")

    monkeypatch.setattr(gpu_client.httpx, "Client", _FailingClient)
    remote = RemoteSam3dReconstructor("http://offline", "token", tmp_path)
    validate_glb(remote.reconstruct(image, mask))
    assert _FailingClient.calls == 1
    assert remote.last_provenance.source == "fallback"


def test_stub_response_is_refused_without_explicit_allowance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image, mask = _inputs()
    _Client.response = _Response(200, StubReconstructor().reconstruct(image, mask), backend="stub")
    monkeypatch.setattr(gpu_client.httpx, "Client", _Client)
    remote = RemoteSam3dReconstructor("http://remote", "token", tmp_path, allow_stub=False)
    remote.reconstruct(image, mask)
    assert remote.last_provenance.source == "fallback"

def test_stub_cache_cannot_cross_into_production_policy(tmp_path,monkeypatch):
    image,mask=_inputs()
    _Client.calls=0
    _Client.response=_Response(200,StubReconstructor().reconstruct(image,mask),backend="stub")
    monkeypatch.setattr(gpu_client.httpx,"Client",_Client)
    allowed=RemoteSam3dReconstructor("http://local-stub","fixture-token",tmp_path,allow_stub=True)
    allowed.reconstruct(image,mask)
    assert allowed.last_provenance.backend=="stub"
    _Client.response=_Response(401,b"")
    production=RemoteSam3dReconstructor("http://remote","fixture-token",tmp_path,allow_stub=False)
    production.reconstruct(image,mask)
    assert production.last_provenance.source=="fallback" and _Client.calls==2

def test_real_backend_cannot_initialize_in_phase_a(monkeypatch):
    from splatsam.recon.sam3d_backend import Sam3dBackend
    from splatsam.recon.gpu_client import ReconstructionError
    monkeypatch.delenv("SPLATSAM_PHASE",raising=False)
    monkeypatch.setenv("SPLATSAM_GPU_APPROVED","1")
    backend=Sam3dBackend()
    assert backend.availability()[0] is False
    with pytest.raises(ReconstructionError,match="Phase B"):
        backend.reconstruct(*_inputs(),seed=42)

"""Operator-confirmed phase gates. These guard mistakes, not a malicious owner."""

from pathlib import Path

from splatsam.utils import ROOT, file_digest, load_json


class AuthorizationError(RuntimeError):
    pass


def require_live(action: str, video: Path | None = None, manifest: Path | None = None) -> dict:
    path = manifest or ROOT / "data" / "authorization.json"
    if not path.exists():
        raise AuthorizationError("Phase B requires user go and an operator-recorded action confirmation")
    auth = load_json(path)
    if auth.get("go") is not True or auth.get("actions", {}).get(action) is not True:
        raise AuthorizationError(f"User confirmation missing for {action}")
    if action in {"B2", "B3", "B4"} and auth.get("masks_approved") is not True:
        raise AuthorizationError("User overlay approval required")
    if video is not None:
        video = video.resolve()
        if not video.is_relative_to((ROOT / "data/videos").resolve()):
            raise AuthorizationError("Only user-provided data/videos media allowed")
        if auth.get("video_sha256") != file_digest(video) or auth.get("privacy_ready") is not True:
            raise AuthorizationError("Input digest/privacy confirmation missing or stale")
    if action == "B3" and auth.get("gpu_spend_approved") is not True:
        raise AuthorizationError("GPU spend requires separate user approval")
    return auth

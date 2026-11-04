"""Canonical artifacts, safe local writes and reproducible hashes."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[2]

def canonical(value: Any) -> str:
    if hasattr(value, 'model_dump'):
        value = value.model_dump(mode='json')
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(canonical(value) + '\n')
    os.replace(temporary, path)

def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())

def config() -> dict:
    return yaml.safe_load((ROOT / 'config.yaml').read_text())

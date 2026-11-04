"""Cached model adapters and typed responses."""

from .client import ModelClient
from .mock import MockClient

__all__ = ["MockClient", "ModelClient"]

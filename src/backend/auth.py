"""Basic API-key auth for /predict (Day 3 task).

Architecture diagram promises an auth layer. This is intentionally simple
(bearer token, not full OAuth) for Phase I.
"""
from __future__ import annotations

import os

from fastapi import Header, HTTPException

API_KEY = os.getenv("CICD_API_KEY", "dev-key-change-me")


def require_api_key(x_api_key: str | None = Header(default=None)):
    if x_api_key is None:
        # Also accept Authorization: Bearer <key>
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


def check_bearer(authorization: str | None) -> bool:
    """Used by Lambda which sends Authorization: Bearer <key>."""
    if not authorization or not authorization.startswith("Bearer "):
        return False
    return authorization.split(" ", 1)[1].strip() == API_KEY

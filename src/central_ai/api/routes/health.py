"""Health check router."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    """Service health verification endpoint."""
    return {"status": "ok", "service": "central-ai"}

"""FastAPI dependencies for authentication and tenancy headers."""

from __future__ import annotations

from typing import Dict
from fastapi import Header, HTTPException, Depends

from ..core.config import API_KEY, DEFAULT_PROJECT_ID, DEFAULT_ORG_ID


def require_api_key(
    x_api_key: str = Header(default="dev-key", alias="X-Api-Key"),
) -> str:
    """Validate incoming API key against configured CENTRAL_API_KEY."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def get_tenant(
    x_project_id: str = Header(default=DEFAULT_PROJECT_ID, alias="X-Project-Id"),
    x_org_id: str = Header(default=DEFAULT_ORG_ID, alias="X-Org-Id"),
    _: str = Depends(require_api_key),
) -> Dict[str, str]:
    """Extract tenancy routing context for multi-product centralization."""
    return {"project_id": x_project_id, "org_id": x_org_id}

from fastapi import APIRouter, Depends
from ..dependencies import get_tenant, require_api_key

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token")
def create_token(_: str = Depends(require_api_key), tenant: dict = Depends(get_tenant)):
    return {
        "access_token": "dev-token",
        "token_type": "bearer",
        "org_id": tenant["org_id"],
        "project_id": tenant["project_id"],
    }


@router.get("/validate")
def validate(_: str = Depends(require_api_key), tenant: dict = Depends(get_tenant)):
    return {"valid": True, **tenant}

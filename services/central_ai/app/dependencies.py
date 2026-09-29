from fastapi import Header, Depends, HTTPException
from sqlalchemy.orm import Session
from . import database, models
from .config import API_KEY


def require_api_key(x_api_key: str = Header(default="dev-key", alias="X-Api-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def get_tenant(
    x_project_id: str = Header(default="hpi", alias="X-Project-Id"),
    x_org_id: str = Header(default="org-default", alias="X-Org-Id"),
    _: str = Depends(require_api_key),
):
    return {"project_id": x_project_id, "org_id": x_org_id}


def get_current_device(
    x_device_id: str = Header(default="web-client", alias="X-Device-ID"),
    db: Session = Depends(database.get_db),
):
    device = db.query(models.Device).filter(models.Device.device_id == x_device_id).first()
    if not device:
        device = models.Device(device_id=x_device_id)
        db.add(device)
        db.commit()
        db.refresh(device)
    return device

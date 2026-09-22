from fastapi import Header, Depends, HTTPException
from sqlalchemy.orm import Session
from . import database, models

def get_current_device(x_device_id: str = Header(...), db: Session = Depends(database.get_db)):
    if not x_device_id:
        raise HTTPException(status_code=400, detail="X-Device-ID header missing")
    
    device = db.query(models.Device).filter(models.Device.device_id == x_device_id).first()
    if not device:
        # Auto-register device
        device = models.Device(device_id=x_device_id)
        db.add(device)
        db.commit()
        db.refresh(device)
        
    return device

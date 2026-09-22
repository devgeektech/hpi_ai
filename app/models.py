from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    uploads = relationship("NoteUpload", back_populates="device")

class NoteUpload(Base):
    __tablename__ = "note_uploads"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey("devices.id"))
    filename = Column(String)
    scan_quality = Column(String)
    handwriting_detected = Column(Float)
    retake = Column(Boolean)
    detected_text = Column(String)
    profile_data = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    device = relationship("Device", back_populates="uploads")

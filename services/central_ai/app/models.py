from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class Organization(Base):
    __tablename__ = "organizations"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(String, unique=True, index=True)
    name = Column(String, default="Default Org")
    created_at = Column(DateTime, default=datetime.utcnow)


class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(String, unique=True, index=True)  # hmb | boardroom | hpi
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class Device(Base):
    __tablename__ = "devices"
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    uploads = relationship("NoteUpload", back_populates="device")


class NoteUpload(Base):
    """Legacy-compatible upload row kept for demo continuity."""
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


class Submission(Base):
    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True, index=True)
    public_id = Column(String, unique=True, index=True)
    org_id = Column(String, index=True)
    project_id = Column(String, index=True)
    device_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ImageAsset(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    public_id = Column(String, unique=True, index=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=True)
    filename = Column(String)
    path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class AIJob(Base):
    __tablename__ = "ai_jobs"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    job_type = Column(String, default="handwriting_analyze")
    status = Column(String, default="queued", index=True)
    org_id = Column(String, index=True)
    project_id = Column(String, index=True)
    image_path = Column(String, nullable=True)
    result_id = Column(String, nullable=True)
    message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(String, unique=True, index=True)
    job_id = Column(String, index=True)
    status = Column(String, default="completed")  # completed | needs_review | approved | rejected
    features_json = Column(Text)
    profile_json = Column(Text)
    confidence = Column(Float, default=0.0)
    model_version = Column(String)
    methodology_version = Column(String)
    rule_version = Column(String)
    feature_schema_version = Column(String)
    report_version = Column(String, default="report-v1")
    created_at = Column(DateTime, default=datetime.utcnow)


class ReviewTask(Base):
    __tablename__ = "review_tasks"
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    result_id = Column(String, index=True)
    status = Column(String, default="pending")  # pending | approved | rejected | revised
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    decided_at = Column(DateTime, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    action = Column(String)
    entity_type = Column(String)
    entity_id = Column(String)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

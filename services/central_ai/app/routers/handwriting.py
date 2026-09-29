import json
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile
from sqlalchemy.orm import Session

from .. import database, models
from ..config import UPLOADS_DIR
from ..dependencies import get_tenant, get_current_device
from ..engines.cv import analyze_image_quality, perform_advanced_analysis
from ..jobs.runner import run_handwriting_job

router = APIRouter(prefix="/handwriting", tags=["handwriting"])


def _process_job(job_id: str):
    db = database.SessionLocal()
    try:
        run_handwriting_job(db, job_id)
    finally:
        db.close()


@router.post("/analyze")
async def analyze_handwriting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(database.get_db),
    tenant: dict = Depends(get_tenant),
    device: models.Device = Depends(get_current_device),
):
    data = await file.read()
    job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"
    dest = UPLOADS_DIR / f"{job_id}_{file.filename or 'note.jpg'}"
    dest.write_bytes(data)

    job = models.AIJob(
        job_id=job_id,
        job_type="handwriting_analyze",
        status="queued",
        org_id=tenant["org_id"],
        project_id=tenant["project_id"],
        image_path=str(dest),
        message="Queued",
    )
    db.add(job)
    db.commit()

    background_tasks.add_task(_process_job, job_id)
    return {"job_id": job_id, "status": "queued"}


@router.post("/quality")
async def handwriting_quality(file: UploadFile = File(...), tenant: dict = Depends(get_tenant)):
    data = await file.read()
    return analyze_image_quality(data)


@router.post("/features")
async def handwriting_features(file: UploadFile = File(...), tenant: dict = Depends(get_tenant)):
    data = await file.read()
    return perform_advanced_analysis(data)

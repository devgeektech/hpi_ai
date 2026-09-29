import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from .. import database, models
from ..config import UPLOADS_DIR
from ..dependencies import get_tenant
from ..engines.cv import analyze_image_quality

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(database.get_db),
    tenant: dict = Depends(get_tenant),
):
    data = await file.read()
    public_id = f"IMG-{uuid.uuid4().hex[:10].upper()}"
    dest = UPLOADS_DIR / f"{public_id}_{file.filename or 'upload.jpg'}"
    dest.write_bytes(data)

    submission = models.Submission(
        public_id=f"SUB-{uuid.uuid4().hex[:8].upper()}",
        org_id=tenant["org_id"],
        project_id=tenant["project_id"],
    )
    db.add(submission)
    db.flush()

    asset = models.ImageAsset(
        public_id=public_id,
        submission_id=submission.id,
        filename=file.filename,
        path=str(dest),
    )
    db.add(asset)
    db.commit()
    return {"image_id": public_id, "submission_id": submission.public_id, "path": str(dest)}


@router.post("/validate")
async def validate_image(
    file: UploadFile = File(...),
    tenant: dict = Depends(get_tenant),
):
    data = await file.read()
    result = analyze_image_quality(data)
    return result


@router.post("/preprocess")
async def preprocess_image(file: UploadFile = File(...), tenant: dict = Depends(get_tenant)):
    # Placeholder: quality check only for MVP; full warp happens inside OCR
    data = await file.read()
    quality = analyze_image_quality(data)
    return {"status": "ok", "quality": quality, "steps": ["normalize", "ready_for_model"]}

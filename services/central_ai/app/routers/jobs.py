import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, database
from ..dependencies import get_tenant

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}")
def get_job(job_id: str, db: Session = Depends(database.get_db), tenant: dict = Depends(get_tenant)):
    job = db.query(models.AIJob).filter(models.AIJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "result_id": job.result_id,
        "message": job.message,
        "project_id": job.project_id,
    }


@router.get("/{job_id}/result")
def get_job_result(job_id: str, db: Session = Depends(database.get_db), tenant: dict = Depends(get_tenant)):
    job = db.query(models.AIJob).filter(models.AIJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result_id:
        raise HTTPException(status_code=404, detail="Result not ready")
    result = (
        db.query(models.AnalysisResult)
        .filter(models.AnalysisResult.result_id == job.result_id)
        .first()
    )
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    features = json.loads(result.features_json or "{}")
    profile = json.loads(result.profile_json or "{}")
    return {
        "result_id": result.result_id,
        "job_id": result.job_id,
        "status": result.status,
        "features": features,
        "profile": profile,
        "confidence": result.confidence,
        "model_version": result.model_version,
        "methodology_version": result.methodology_version,
        "rule_version": result.rule_version,
        "feature_schema_version": result.feature_schema_version,
        "report_version": result.report_version,
    }

import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import database, models
from ..dependencies import get_tenant

router = APIRouter(prefix="/hpi", tags=["hpi"])


def _get_result(db: Session, result_id: str) -> models.AnalysisResult:
    result = (
        db.query(models.AnalysisResult)
        .filter(models.AnalysisResult.result_id == result_id)
        .first()
    )
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


@router.get("/profile/{result_id}")
def get_profile(result_id: str, db: Session = Depends(database.get_db), tenant: dict = Depends(get_tenant)):
    result = _get_result(db, result_id)
    profile = json.loads(result.profile_json or "{}")
    return {
        "result_id": result_id,
        "status": result.status,
        "confidence": result.confidence,
        "profile": profile,
        "methodology_version": result.methodology_version,
        "rule_version": result.rule_version,
    }


@router.get("/traits/{result_id}")
def get_traits(result_id: str, db: Session = Depends(database.get_db), tenant: dict = Depends(get_tenant)):
    result = _get_result(db, result_id)
    profile = json.loads(result.profile_json or "{}")
    return {"result_id": result_id, "scores": profile.get("scores", []), "insights": profile.get("insights", [])}


@router.get("/patterns/{result_id}")
def get_patterns(result_id: str, db: Session = Depends(database.get_db), tenant: dict = Depends(get_tenant)):
    result = _get_result(db, result_id)
    profile = json.loads(result.profile_json or "{}")
    return {
        "result_id": result_id,
        "standout_strength": profile.get("standout_strength"),
        "patterns": [i.get("title") for i in profile.get("insights", [])],
    }


@router.get("/confidence/{result_id}")
def get_confidence(result_id: str, db: Session = Depends(database.get_db), tenant: dict = Depends(get_tenant)):
    result = _get_result(db, result_id)
    return {"result_id": result_id, "confidence": result.confidence, "status": result.status}


@router.post("/analyze")
def analyze_alias():
    return {
        "message": "Use POST /api/v1/handwriting/analyze then poll /api/v1/jobs/{job_id}",
    }

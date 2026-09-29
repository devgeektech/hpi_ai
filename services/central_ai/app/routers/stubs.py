from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Any, Dict, Optional

from ..dependencies import get_tenant
from ..engines.nlp import explain

router = APIRouter(tags=["product-stubs"])


class NlpRequest(BaseModel):
    result_id: Optional[str] = None
    standout_strength: Optional[str] = None
    confidence: float = 0
    status: str = "approved"
    payload: Dict[str, Any] = {}


@router.post("/hmb/evaluate-exercise")
def hmb_evaluate(tenant: dict = Depends(get_tenant)):
    return {
        "project": "hmb",
        "skill_score": 82,
        "strengths": ["Letter formation consistency"],
        "improvements": ["Baseline alignment"],
        "message": "Stub — wire exercise rubric next",
    }


@router.post("/hmb/handwriting-feedback")
def hmb_feedback(tenant: dict = Depends(get_tenant)):
    return {"feedback": "Keep spacing even between words.", "mastery_delta": 0.05}


@router.get("/hmb/mastery")
def hmb_mastery(tenant: dict = Depends(get_tenant)):
    return {"mastery": 0.64, "next_exercise_id": "ex-baseline-01"}


@router.get("/hmb/next-exercise")
def hmb_next(tenant: dict = Depends(get_tenant)):
    return {"exercise_id": "ex-baseline-01", "title": "Baseline practice"}


@router.post("/boardroom/participant-analysis")
def br_participant(tenant: dict = Depends(get_tenant)):
    return {"message": "Submit handwriting via /handwriting/analyze then attach result_id to participant"}


@router.post("/boardroom/team-analysis")
def br_team(tenant: dict = Depends(get_tenant)):
    return {"team_patterns": [], "message": "Stub — aggregate participant result_ids"}


@router.get("/boardroom/team-patterns")
def br_patterns(tenant: dict = Depends(get_tenant)):
    return {"patterns": ["Complementary decision styles"], "gaps": ["Shared focus risk"]}


@router.post("/boardroom/team-report")
def br_report(tenant: dict = Depends(get_tenant)):
    return {"report_type": "team", "status": "draft"}


@router.post("/reports/generate")
def reports_generate(body: Dict[str, Any], tenant: dict = Depends(get_tenant)):
    report_type = body.get("report_type", "hpi_profile")
    return {
        "report_id": f"RPT-{report_type}",
        "report_type": report_type,
        "status": "draft",
        "report_version": "report-v1",
    }


@router.post("/nlp/explain")
def nlp_explain(body: NlpRequest, tenant: dict = Depends(get_tenant)):
    if body.status not in ("approved", "completed"):
        return {"error": "NLP only allowed on approved results", "allowed": False}
    text = explain(
        {
            "standout_strength": body.standout_strength or body.payload.get("standout_strength"),
            "confidence": body.confidence,
        }
    )
    return {"allowed": True, **text}

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import database, models
from ..dependencies import get_tenant

router = APIRouter(prefix="/reviews", tags=["reviews"])


class ReviewDecision(BaseModel):
    notes: str = ""


@router.get("")
def list_reviews(db: Session = Depends(database.get_db), tenant: dict = Depends(get_tenant)):
    tasks = db.query(models.ReviewTask).order_by(models.ReviewTask.created_at.desc()).limit(50).all()
    return [
        {
            "task_id": t.task_id,
            "result_id": t.result_id,
            "status": t.status,
            "notes": t.notes,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in tasks
    ]


def _decide(db: Session, task_id: str, status: str, notes: str):
    task = db.query(models.ReviewTask).filter(models.ReviewTask.task_id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Review task not found")
    task.status = status
    task.notes = notes
    task.decided_at = datetime.utcnow()

    result = (
        db.query(models.AnalysisResult)
        .filter(models.AnalysisResult.result_id == task.result_id)
        .first()
    )
    if result:
        result.status = "approved" if status == "approved" else status

    job = db.query(models.AIJob).filter(models.AIJob.result_id == task.result_id).first()
    if job and status == "approved":
        job.status = "completed"
        job.message = "Approved by reviewer"

    db.add(
        models.AuditLog(
            action=f"review_{status}",
            entity_type="review_task",
            entity_id=task_id,
            detail=notes,
        )
    )
    db.commit()
    return {"task_id": task_id, "status": status}


@router.post("/{task_id}/approve")
def approve(task_id: str, body: ReviewDecision = ReviewDecision(), db: Session = Depends(database.get_db), tenant: dict = Depends(get_tenant)):
    return _decide(db, task_id, "approved", body.notes)


@router.post("/{task_id}/reject")
def reject(task_id: str, body: ReviewDecision = ReviewDecision(), db: Session = Depends(database.get_db), tenant: dict = Depends(get_tenant)):
    return _decide(db, task_id, "rejected", body.notes)


@router.post("/{task_id}/override")
def override(task_id: str, body: ReviewDecision = ReviewDecision(), db: Session = Depends(database.get_db), tenant: dict = Depends(get_tenant)):
    return _decide(db, task_id, "approved", f"OVERRIDE: {body.notes}")

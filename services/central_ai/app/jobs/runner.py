import json
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from .. import models
from ..config import CONFIDENCE_AUTO_APPROVE
from ..engines.cv import process_handwritten_note, perform_advanced_analysis
from ..engines.hpi import generate_profile

try:
    from shared.versions import VERSIONS
except ImportError:
    VERSIONS = {
        "model_version": "handwriting-v1",
        "methodology_version": "hpi-v1",
        "rule_version": "rules-v1",
        "feature_schema_version": "features-v1",
        "report_version": "report-v1",
    }


def run_handwriting_job(db: Session, job_id: str) -> None:
    job = db.query(models.AIJob).filter(models.AIJob.job_id == job_id).first()
    if not job:
        return

    job.status = "processing"
    job.message = "Running handwriting analysis"
    job.updated_at = datetime.utcnow()
    db.commit()

    try:
        image_path = Path(job.image_path)
        image_bytes = image_path.read_bytes()

        ocr = process_handwritten_note(image_bytes)
        advanced = perform_advanced_analysis(image_bytes)
        profile = generate_profile(ocr.get("boxes", []), ocr.get("detected_text", ""))

        # Confidence heuristic from handwriting_detected + retake
        confidence = float(ocr.get("handwriting_detected") or 0.0)
        if ocr.get("retake"):
            confidence = min(confidence, 40.0)

        features = {
            **advanced,
            "detected_text": ocr.get("detected_text", ""),
            "scan_quality": ocr.get("scan_quality", "unknown"),
            "handwriting_detected": confidence,
            "retake": bool(ocr.get("retake")),
            "numeric": {
                "handwriting_detected": confidence,
            },
        }

        result_id = f"RESULT-{uuid.uuid4().hex[:8].upper()}"
        status = "approved" if confidence >= CONFIDENCE_AUTO_APPROVE else "needs_review"

        result = models.AnalysisResult(
            result_id=result_id,
            job_id=job_id,
            status=status,
            features_json=json.dumps(features),
            profile_json=json.dumps(profile),
            confidence=confidence,
            model_version=VERSIONS["model_version"],
            methodology_version=VERSIONS["methodology_version"],
            rule_version=VERSIONS["rule_version"],
            feature_schema_version=VERSIONS["feature_schema_version"],
            report_version=VERSIONS.get("report_version", "report-v1"),
        )
        db.add(result)

        if status == "needs_review":
            task = models.ReviewTask(
                task_id=f"REV-{uuid.uuid4().hex[:8].upper()}",
                result_id=result_id,
                status="pending",
            )
            db.add(task)
            job.status = "needs_review"
            job.message = "Low confidence — awaiting human review"
        else:
            job.status = "completed"
            job.message = "Analysis complete (auto-approved)"

        job.result_id = result_id
        job.updated_at = datetime.utcnow()
        db.add(
            models.AuditLog(
                action="job_completed",
                entity_type="ai_job",
                entity_id=job_id,
                detail=json.dumps({"result_id": result_id, "status": status}),
            )
        )
        db.commit()
    except Exception as exc:
        job.status = "failed"
        job.message = str(exc)
        job.updated_at = datetime.utcnow()
        db.commit()

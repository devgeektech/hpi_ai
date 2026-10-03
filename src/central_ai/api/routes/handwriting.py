"""Handwriting analysis and assessment router."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from central_ai.api.dependencies import get_tenant
from central_ai.core.config import UPLOADS_DIR, VERSIONS
from central_ai.core.logging import logger
from central_ai.engines.cv_preprocessing import perform_advanced_analysis
from central_ai.engines.ocr_engine import process_handwritten_note
from central_ai.engines.hpi_assessment import (
    generate_profile,
    get_model_status,
    record_and_train_sample,
)
from central_ai.schemas.handwriting import (
    AnalysisResultPayload,
    FeatureSet,
    AssessmentProfile,
    ModelStatusResponse,
    TrainingResultResponse,
)

router = APIRouter(prefix="/handwriting", tags=["handwriting"])


@router.post("/analyze", response_model=AnalysisResultPayload)
async def analyze_handwriting(
    file: UploadFile = File(...),
    engine: str = Query(
        default="paddle",
        description="OCR recognition engine: 'paddle' (fast, exact, universal cursive/Latin/English) or 'trocr' (deep English transformer)",
    ),
    verified_scores: Optional[str] = Form(
        default=None,
        description="Optional JSON string of verified HPI trait scores (e.g. '{\"Execution\": 85, \"Focus\": 90}') to train the model side-by-side with this upload.",
    ),
    tenant: dict = Depends(get_tenant),
):
    """Stateless handwriting analysis endpoint:

    Performs OCR, extracts handwriting geometry, and generates the HPI assessment profile.
    If verified_scores is supplied, trains the machine learning model side-by-side in real time!
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "note.jpg"
    result_id = f"RESULT-{uuid.uuid4().hex[:8].upper()}"
    destination_path = UPLOADS_DIR / f"{result_id}_{filename}"

    try:
        destination_path.write_bytes(data)
    except Exception as exc:
        logger.warning(f"Could not persist upload locally: {exc}")

    pipeline_start = time.perf_counter()
    logger.info(
        f"[{result_id}] Starting handwriting analysis pipeline for project '{tenant['project_id']}' "
        f"({len(data):,} bytes, engine='{engine}')"
    )

    try:
        ocr_result = process_handwritten_note(data, engine=engine)

        t_geom_start = time.perf_counter()
        logger.info(f"[{result_id}] [Step 3/4] Extracting handwriting stroke geometry & pressure...")
        advanced_features = perform_advanced_analysis(data)
        logger.info(f"[{result_id}] [Step 3/4] Geometry analysis completed in {time.perf_counter() - t_geom_start:.2f}s")

        # Side-by-side active training if verified scores accompany the upload
        training_info = None
        if verified_scores:
            try:
                parsed_scores = json.loads(verified_scores)
                if isinstance(parsed_scores, dict):
                    training_info = record_and_train_sample(
                        features=advanced_features,
                        verified_scores=parsed_scores,
                        sample_id=result_id,
                    )
                    logger.info(
                        f"[{result_id}] [Active Learning] Trained model side-by-side with upload! "
                        f"Status: {training_info.get('status')}, Total samples: {training_info.get('total_samples_trained')}"
                    )
            except Exception as train_exc:
                logger.warning(f"[{result_id}] Side-by-side training skipped: {train_exc}")

        t_hpi_start = time.perf_counter()
        logger.info(f"[{result_id}] [Step 4/4] Generating HPI personality assessment profile...")
        assessment_profile = generate_profile(
            boxes=ocr_result.get("boxes", []),
            text=ocr_result.get("detected_text", ""),
            features=advanced_features,
        )
        logger.info(f"[{result_id}] [Step 4/4] HPI profile generated in {time.perf_counter() - t_hpi_start:.3f}s")

        total_elapsed = time.perf_counter() - pipeline_start
        logger.info(f"[{result_id}] >>> Handwriting analysis completed in {total_elapsed:.2f}s <<<")

        # Formatted console output for side-by-side verification with uploaded image
        extracted_text = ocr_result.get("detected_text", "").strip()
        lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]

        summary_lines = [
            "",
            "=" * 72,
            f"          FINAL EXTRACTED TEXT (IMAGE: {filename})",
            "=" * 72,
            f"Result ID    : {result_id}",
            f"Engine Used  : {ocr_result.get('engine', engine).upper()}",
            f"Image Size   : {len(data):,} bytes",
            f"Total Lines  : {len(lines)} line(s)",
            f"Characters   : {len(extracted_text):,} characters",
            f"Confidence   : {float(ocr_result.get('handwriting_detected') or 0.0):.1f}%",
            f"Standout     : {assessment_profile.get('standout_strength', 'N/A')}",
            f"Elapsed Time : {total_elapsed:.2f}s",
            "-" * 72,
            "LINE-BY-LINE TRANSCRIPTION (Compare directly with uploaded image):",
            "-" * 72,
        ]
        if lines:
            for idx, text_line in enumerate(lines, start=1):
                summary_lines.append(f"  [{idx:02d}] {text_line}")
        else:
            summary_lines.append("  (No handwritten text recognized)")
        summary_lines.append("=" * 72)
        summary_lines.append("")

        logger.info("\n".join(summary_lines))
    except Exception as exc:
        logger.error(f"Analysis engine failure: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Handwriting analysis failed: {exc}") from exc

    confidence = float(ocr_result.get("handwriting_detected") or 0.0)
    if ocr_result.get("retake"):
        confidence = min(confidence, 40.0)

    feature_set = FeatureSet(
        slant=advanced_features.get("slant", "Unknown"),
        spacing=advanced_features.get("spacing", "Unknown"),
        shapes=advanced_features.get("shapes", "Unknown"),
        relative_size=advanced_features.get("relative_size", "Unknown"),
        consistency=advanced_features.get("consistency", "Unknown"),
        stroke_geometry=advanced_features.get("stroke_geometry", "Unknown"),
        detected_text=ocr_result.get("detected_text", ""),
        scan_quality=ocr_result.get("scan_quality", "unknown"),
        handwriting_detected=confidence,
        retake=bool(ocr_result.get("retake")),
    )

    status = "completed" if confidence >= 45.0 else "low_confidence"

    return AnalysisResultPayload(
        result_id=result_id,
        status=status,
        confidence=confidence,
        project_id=tenant["project_id"],
        features=feature_set,
        profile=AssessmentProfile(**assessment_profile),
        model_version=VERSIONS["model_version"],
        methodology_version=VERSIONS["methodology_version"],
        rule_version=VERSIONS["rule_version"],
        feature_schema_version=VERSIONS["feature_schema_version"],
        analyzed_at=datetime.now(timezone.utc).isoformat(),
        upload_path=str(destination_path),
        training_status=training_info,
    )


@router.post("/train", response_model=TrainingResultResponse)
async def train_model_sample(
    file: UploadFile = File(..., description="Handwriting image to train the model with"),
    scores: str = Form(
        ...,
        description="JSON dictionary of verified HPI trait scores, e.g. '{\"Execution\": 85, \"Focus\": 90, \"Logic\": 82}'",
    ),
    tenant: dict = Depends(get_tenant),
):
    """Side-by-side active learning training endpoint:

    Extracts handwriting biometric feature vector from the uploaded image and immediately
    retrains the HPI machine-learning behavioral model.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        scores_dict = json.loads(scores)
        if not isinstance(scores_dict, dict):
            raise ValueError("Scores must be a JSON object mapping trait names to numbers.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid scores JSON: {exc}")

    sample_id = f"TRAIN-{uuid.uuid4().hex[:8].upper()}"
    filename = file.filename or "sample.jpg"

    advanced_features = perform_advanced_analysis(data)
    train_res = record_and_train_sample(
        features=advanced_features,
        verified_scores=scores_dict,
        sample_id=sample_id,
    )

    metrics = (advanced_features.get("metrics") or {})
    numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}

    logger.info(
        f"[{sample_id}] Successfully recorded training sample '{filename}'. "
        f"Model retrained across {train_res.get('total_samples_trained', 0)} sample(s)."
    )

    return TrainingResultResponse(
        status=train_res.get("status", "trained"),
        sample_id=sample_id,
        total_samples_trained=train_res.get("total_samples_trained", 0),
        updated_at=train_res.get("updated_at"),
        recorded_scores=scores_dict,
        metrics_summary=numeric_metrics,
    )


@router.get("/model", response_model=ModelStatusResponse)
async def get_handwriting_model():
    """Inspect the active learning model state, total trained sample count, and active version."""
    status = get_model_status()
    return ModelStatusResponse(**status)


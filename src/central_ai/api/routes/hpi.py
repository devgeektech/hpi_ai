"""HPI Individual API Router.

Dedicated project router for HPI methodology, findings, PRIME™, and ALIGN™.
Uses the physical parameters and metrics already extracted by the handwriting perception core.
No duplicate image uploads or redundant CV processing.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException

from central_ai.api.dependencies import get_tenant
from central_ai.core.logging import logger
from central_ai.engines.hpi.methodology_engine import (
    evaluate_hpi_methodology,
    load_methodology_rules,
)
from central_ai.schemas.hpi import (
    HPIAssessmentResult,
    HPIAssessRequest,
)

router = APIRouter(prefix="/hpi", tags=["hpi"])


@router.post(
    "/assess",
    response_model=HPIAssessmentResult,
    summary="Evaluate HPI methodology directly from extracted handwriting parameters",
    description="""Runs client-approved HPI methodology engine directly on previously extracted parameters.

Accepts either:
1. The full response dictionary from POST /api/v1/handwriting/analyze
2. Or a direct metrics dictionary ({'slant_angle': 2.5, 'pressure_density': 0.08, ...})

Uses already analyzed parameters without re-uploading the image.
""",
)
async def assess_hpi_parameters(
    payload: HPIAssessRequest,
    tenant: dict = Depends(get_tenant),
):
    """Assess previously extracted handwriting biometric parameters."""
    result_id = payload.result_id or payload.sample_id or f"HPI-{uuid.uuid4().hex[:8].upper()}"

    # Extract metrics from either direct metrics field or handwriting_features block
    metrics: Dict[str, float] = {}
    if payload.metrics:
        metrics = {k: float(v) for k, v in payload.metrics.items() if isinstance(v, (int, float))}
    elif payload.handwriting_features and "metrics" in payload.handwriting_features:
        raw_m = payload.handwriting_features["metrics"]
        if isinstance(raw_m, dict):
            metrics = {k: float(v) for k, v in raw_m.items() if isinstance(v, (int, float))}

    if not metrics:
        raise HTTPException(
            status_code=400,
            detail="No valid handwriting feature metrics provided. Please provide 'metrics' or 'handwriting_features.metrics'.",
        )

    # Check if quality gate from handwriting/analyze was passed
    q_gate = payload.quality_gate or {}
    if q_gate.get("rescan_required"):
        issues = q_gate.get("issues", [])
        explanation = "; ".join(f"[{i.get('criterion', '').title()}] {i.get('message', '')}" for i in issues)
        failed_page = q_gate.get("failed_page", 1)
        logger.warning(f"[{result_id}] Rejecting HPI assessment due to failed image quality gate.")
        return HPIAssessmentResult(
            result_id=result_id,
            status="failed_quality_check",
            review_status="in_human_review",
            rescan_required=True,
            standout_strength="N/A",
            findings=[],
            prime=None,
            align=None,
            confidence_vector={
                "quality_confidence": 0.0,
                "detection_confidence": 0.0,
                "ocr_confidence": 0.0,
                "feature_confidence": 0.0,
                "methodology_confidence": 0.0,
                "overall_confidence": 0.0,
            },
            methodology_version="2.4.0",
            rule_version="2026.1",
            analyzed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            project_id=tenant.get("project_id", "hpi"),
            message=f"Image quality requirements failed. Please retake/rescan page {failed_page}. Issues: {explanation}",
        )

    # Determine confidence levels
    ocr_conf = payload.ocr_confidence
    if ocr_conf is None and payload.ocr:
        ocr_conf = float(payload.ocr.get("ocr_confidence", 95.0))
    if ocr_conf is None:
        ocr_conf = 95.0

    quality_conf = payload.quality_confidence
    if quality_conf is None and q_gate:
        quality_conf = float(q_gate.get("metrics", {}).get("quality_confidence", 90.0))
    if quality_conf is None:
        quality_conf = 90.0

    quality_report = {
        "overall_status": q_gate.get("overall_status", "Good"),
        "metrics": {"quality_confidence": quality_conf},
    }

    # Evaluate HPI methodology reasoning
    t_start = time.perf_counter()
    assessment = evaluate_hpi_methodology(
        metrics=metrics,
        result_id=result_id,
        quality_report=quality_report,
        ocr_confidence=ocr_conf,
        project_id=tenant.get("project_id", "hpi"),
    )

    logger.info(
        f"[{result_id}] Assessed HPI methodology from extracted parameters in {time.perf_counter() - t_start:.3f}s "
        f"(Standout: {assessment.standout_strength}, Review: {assessment.review_status})"
    )
    return assessment


@router.get(
    "/rules",
    summary="Inspect client-approved HPI methodology rules catalog",
    description="Returns the active HPI methodology rules registry, trait definitions, and benchmark thresholds.",
)
async def get_methodology_rules(
    tenant: dict = Depends(get_tenant),
):
    """Return active methodology rules catalog."""
    return load_methodology_rules()

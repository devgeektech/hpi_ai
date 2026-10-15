"""Active Learning & Model Training Router.

Dedicated endpoints for submitting verified handwriting training samples
and inspecting active learning model weights and versions.
"""

from __future__ import annotations

import json
import uuid
from typing import Dict
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from central_ai.api.dependencies import get_tenant
from central_ai.core.logging import logger
from central_ai.engines.active_learning import (
    get_model_status,
    record_and_train_sample,
)
from central_ai.engines.cv_preprocessing import perform_advanced_analysis
from central_ai.schemas.handwriting import (
    ModelStatusResponse,
    TrainingResultResponse,
)

router = APIRouter(prefix="/training", tags=["training"])


@router.post(
    "/train",
    response_model=TrainingResultResponse,
    summary="Train Model Sample",
    description="""Submit a verified handwriting sample with ground-truth trait scores to immediately update the Ridge regression model weights.""",
)
async def train_model_sample(
    file: UploadFile = File(..., description="Handwriting image to train the model with"),
    scores: str = Form(
        ...,
        description="JSON dictionary of verified scores, e.g. '{\"Execution\": 85, \"Focus\": 90, \"Logic\": 82}'",
    ),
    tenant: dict = Depends(get_tenant),
):
    """Side-by-side active learning training endpoint."""
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


@router.get(
    "/model",
    response_model=ModelStatusResponse,
    summary="Get Handwriting Model",
    description="Inspect the active learning model state, total trained sample count, and active version.",
)
async def get_handwriting_model():
    """Inspect active learning model state."""
    status = get_model_status()
    return ModelStatusResponse(**status)

"""Image quality validation router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from central_ai.api.dependencies import get_tenant
from central_ai.core.logging import logger
from central_ai.engines.cv_preprocessing import decode_image_bytes, evaluate_scan_quality
from central_ai.engines.ocr_engine import detect_text_boxes
from central_ai.schemas.handwriting import ImageQualityResponse

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/validate", response_model=ImageQualityResponse)
async def validate_image(
    file: UploadFile = File(...),
    tenant: dict = Depends(get_tenant),
):
    """Validate uploaded image quality and handwriting presence."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.debug(f"Validating image '{file.filename}' ({len(data)} bytes) for tenant: {tenant['project_id']}")

    img_cv = decode_image_bytes(data)
    if img_cv is None or img_cv.size == 0:
        return ImageQualityResponse(
            scan_quality="poor",
            handwriting_detected=0.0,
            retake=True,
        )

    scan_quality, variance = evaluate_scan_quality(img_cv)
    boxes = detect_text_boxes(img_cv)

    if not boxes:
        return ImageQualityResponse(
            scan_quality=scan_quality,
            handwriting_detected=0.0,
            retake=True,
        )

    total_chars = sum(
        len(box[1][0])
        for box in boxes
        if len(box) > 1 and box[1] and len(box[1]) > 0 and isinstance(box[1][0], str)
    )

    if total_chars < 35:
        handwriting_detected = max(10.0, float(total_chars))
    else:
        handwriting_detected = min(75.0 + (total_chars * 0.2), 99.9)

    retake = handwriting_detected < 45.0 or scan_quality == "poor"

    return ImageQualityResponse(
        scan_quality=scan_quality,
        handwriting_detected=float(round(handwriting_detected, 1)),
        retake=retake,
    )

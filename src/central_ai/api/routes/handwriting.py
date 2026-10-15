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
from central_ai.engines.cv_preprocessing import (
    decode_image_bytes,
    evaluate_scan_quality,
    perform_advanced_analysis,
)
from central_ai.engines.ocr_engine import process_handwritten_note
from central_ai.schemas.handwriting import (
    AnalysisMetadata,
    AnalysisResultPayload,
    FeatureSet,
    HandwritingFeaturesResult,
    ImageDimension,
    OcrExtractionResult,
    OcrLineItem,
    QualityGateResult,
)

router = APIRouter(prefix="/handwriting", tags=["handwriting"])


@router.post(
    "/analyze",
    response_model=AnalysisResultPayload,
    summary="check image quality gate and extract text on it",
    description="""Before performing HPI analysis, evaluate the submitted handwriting image.

Check for:

- Blur
- Glare
- Cropping
- Perspective problems
- Incomplete page
- Whether handwriting is actually visible
- Whether the page is usable for analysis

Classify the image as:

- Good
- Warning
- Failed

If the image fails quality requirements:

- Do not continue with HPI analysis.
- Clearly explain what is wrong.
- Ask the user to retake/rescan only the required page.
- Do not generate personality or behavioral findings from a failed image.

The HPI specification requires image quality gating before analysis.""",
)
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
    """Before performing HPI analysis, evaluate the submitted handwriting image.

    Check for:

    - Blur
    - Glare
    - Cropping
    - Perspective problems
    - Incomplete page
    - Whether handwriting is actually visible
    - Whether the page is usable for analysis

    Classify the image as:

    - Good
    - Warning
    - Failed

    If the image fails quality requirements:

    - Do not continue with HPI analysis.
    - Clearly explain what is wrong.
    - Ask the user to retake/rescan only the required page.
    - Do not generate personality or behavioral findings from a failed image.

    The HPI specification requires image quality gating before analysis.
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
    img_np = decode_image_bytes(data)
    img_h, img_w = (img_np.shape[0], img_np.shape[1]) if img_np is not None else (0, 0)

    logger.info(
        f"[{result_id}] [Step 1/3] Running Image Quality Gate for project '{tenant['project_id']}' "
        f"({len(data):,} bytes, dimensions={img_w}x{img_h})"
    )

    # 1. Image Quality Gate Evaluation (7-point HPI check)
    quality_report = evaluate_scan_quality(data, page_number=1)
    logger.info(
        f"[{result_id}] Quality Gate Result: {quality_report.overall_status} "
        f"(rescan_required={quality_report.rescan_required}, issues={len(quality_report.issues)})"
    )

    # If the image fails quality requirements:
    # - Do not continue with HPI analysis.
    # - Clearly explain what is wrong.
    # - Ask the user to retake/rescan only the required page.
    # - Do not generate personality or behavioral findings from a failed image.
    if quality_report.rescan_required:
        explanation = "; ".join(f"[{iss['criterion'].title()}] {iss['message']}" for iss in quality_report.issues)
        logger.warning(
            f"[{result_id}] Image failed quality requirements ({quality_report.overall_status}). "
            f"Aborting analysis. Issues: {explanation}"
        )
        return AnalysisResultPayload(
            result_id=result_id,
            status="failed_quality_check",
            message=f"Image quality check failed ({explanation}). Please retake/rescan page {quality_report.failed_page or 1}.",
            quality_gate=QualityGateResult(**quality_report.to_dict()),
            ocr=OcrExtractionResult(
                detected_text="",
                total_lines=0,
                total_words=0,
                total_characters=0,
                ocr_confidence=0.0,
                engine=engine,
                lines=[],
            ),
            handwriting_features=HandwritingFeaturesResult(
                slant="Unknown",
                spacing="Unknown",
                shapes="Unknown",
                relative_size="Unknown",
                consistency="Unknown",
                stroke_geometry="Unknown",
                metrics={},
            ),
            metadata=AnalysisMetadata(
                filename=filename,
                image_size_bytes=len(data),
                dimensions=ImageDimension(width=img_w, height=img_h),
                project_id=tenant["project_id"],
                analyzed_at=datetime.now(timezone.utc).isoformat(),
                processing_time_seconds=round(time.perf_counter() - pipeline_start, 3),
            ),
        )

    logger.info(
        f"[{result_id}] [Step 2/3] Extracting text via OCR engine '{engine}'..."
    )

    try:
        ocr_result = process_handwritten_note(data, engine=engine)

        t_geom_start = time.perf_counter()
        logger.info(f"[{result_id}] [Step 3/3] Extracting handwriting stroke geometry & measurements...")
        advanced_features = perform_advanced_analysis(data)
        logger.info(f"[{result_id}] [Step 3/3] Geometry analysis completed in {time.perf_counter() - t_geom_start:.2f}s")

        total_elapsed = time.perf_counter() - pipeline_start
        logger.info(f"[{result_id}] >>> Handwriting perception completed in {total_elapsed:.2f}s <<<")

        # Line-by-line transcription details
        raw_boxes = ocr_result.get("boxes", [])
        ocr_lines: List[OcrLineItem] = []
        for idx, b in enumerate(raw_boxes, start=1):
            if len(b) > 1 and b[1] and len(b[1]) > 0:
                txt = str(b[1][0]).strip()
                conf_val = float(b[1][1]) if len(b[1]) > 1 else None
                conf_pct = round(conf_val * 100, 1) if conf_val is not None and conf_val <= 1.0 else (conf_val or 0.0)
                poly = b[0]
                poly_list = poly.tolist() if hasattr(poly, "tolist") else poly
                ocr_lines.append(
                    OcrLineItem(
                        line_number=idx,
                        text=txt,
                        confidence=conf_pct,
                        bounding_box=poly_list if isinstance(poly_list, list) else None,
                    )
                )

        extracted_text = ocr_result.get("detected_text", "").strip()
        lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]
        words = extracted_text.split()
        ocr_confidence = float(ocr_result.get("handwriting_detected") or 0.0)

        # Formatted console output for side-by-side verification with uploaded image
        summary_lines = [
            "",
            "=" * 72,
            f"          FINAL EXTRACTED TEXT (IMAGE: {filename})",
            "=" * 72,
            f"Result ID    : {result_id}",
            f"Engine Used  : {ocr_result.get('engine', engine).upper()}",
            f"Image Size   : {len(data):,} bytes ({img_w}x{img_h})",
            f"Quality Gate : {quality_report.overall_status}",
            f"Total Lines  : {len(lines)} line(s)",
            f"Total Words  : {len(words)} word(s)",
            f"Characters   : {len(extracted_text):,} characters",
            f"Confidence   : {ocr_confidence:.1f}%",
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

    status = "completed" if quality_report.overall_status == "Good" else "warning"
    pass_msg = (
        "Image quality check passed. Text extracted successfully."
        if quality_report.overall_status == "Good"
        else f"Image passed with quality warnings: {'; '.join(i['message'] for i in quality_report.issues)}. Text extracted successfully."
    )

    ocr_payload = OcrExtractionResult(
        detected_text=extracted_text,
        total_lines=len(lines),
        total_words=len(words),
        total_characters=len(extracted_text),
        ocr_confidence=ocr_confidence,
        engine=ocr_result.get("engine", engine),
        lines=ocr_lines,
    )

    hw_features_payload = HandwritingFeaturesResult(
        slant=advanced_features.get("slant", "Unknown"),
        spacing=advanced_features.get("spacing", "Unknown"),
        shapes=advanced_features.get("shapes", "Unknown"),
        relative_size=advanced_features.get("relative_size", "Unknown"),
        consistency=advanced_features.get("consistency", "Unknown"),
        stroke_geometry=advanced_features.get("stroke_geometry", "Unknown"),
        metrics=advanced_features.get("metrics", {}),
    )

    metadata_payload = AnalysisMetadata(
        filename=filename,
        image_size_bytes=len(data),
        dimensions=ImageDimension(width=img_w, height=img_h),
        project_id=tenant["project_id"],
        analyzed_at=datetime.now(timezone.utc).isoformat(),
        processing_time_seconds=round(time.perf_counter() - pipeline_start, 3),
    )

    return AnalysisResultPayload(
        result_id=result_id,
        status=status,
        message=pass_msg,
        quality_gate=QualityGateResult(**quality_report.to_dict()),
        ocr=ocr_payload,
        handwriting_features=hw_features_payload,
        metadata=metadata_payload,
    )


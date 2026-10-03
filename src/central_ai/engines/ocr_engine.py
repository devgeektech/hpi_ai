"""Neural OCR text detection (PaddleOCR) and handwritten recognition (PaddleOCR / TrOCR)."""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from PIL import Image

from central_ai.core.config import DEFAULT_OCR_ENGINE, TROCR_MODEL_ID
from central_ai.core.logging import logger
from central_ai.engines.cv_preprocessing import (
    decode_image_bytes,
    evaluate_scan_quality,
    extract_opencv_fallback_boxes,
    four_point_transform,
)

# OpenMP safety for Windows Paddle runtime
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

# Module-level model singletons
_ocr_detector: Optional[Any] = None
_trocr_processor: Optional[Any] = None
_trocr_model: Optional[Any] = None


def get_paddle_ocr() -> Optional[Any]:
    """Lazy initialization of PaddleOCR 2.x text detector."""
    global _ocr_detector
    if _ocr_detector is None:
        try:
            logger.info("Initializing PaddleOCR text detector...")
            from paddleocr import PaddleOCR
            _ocr_detector = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
                use_gpu=False,
                enable_mkldnn=False,
            )
            logger.info("PaddleOCR initialized successfully.")
        except Exception as exc:
            logger.error(f"Failed to initialize PaddleOCR: {exc}")
            _ocr_detector = None
    return _ocr_detector


def get_trocr_models():
    """Lazy initialization of TrOCR processor and vision-encoder-decoder model."""
    global _trocr_processor, _trocr_model
    if _trocr_processor is None or _trocr_model is None:
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            logger.info(f"Loading TrOCR model '{TROCR_MODEL_ID}'...")
            _trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
            _trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _trocr_model = _trocr_model.to(device)
            logger.info(f"TrOCR model loaded successfully on device: {device}.")
        except Exception as exc:
            logger.error(f"Failed to load TrOCR models: {exc}")
            _trocr_processor = None
            _trocr_model = None
    return _trocr_processor, _trocr_model


def detect_text_boxes(img_cv: np.ndarray) -> List[Any]:
    """Run text box detection via PaddleOCR with OpenCV contour fallback."""
    detector = get_paddle_ocr()
    if detector is None:
        logger.warning("PaddleOCR detector unavailable; using OpenCV fallback bounding boxes.")
        return extract_opencv_fallback_boxes(img_cv)

    try:
        raw_result = detector.ocr(img_cv, cls=True)
    except Exception as exc:
        logger.warning(f"PaddleOCR detection failed ({exc}); using OpenCV fallback boxes.")
        return extract_opencv_fallback_boxes(img_cv)

    if not raw_result or not raw_result[0]:
        return []

    page = raw_result[0]
    if isinstance(page, list):
        return page

    # Handle dictionary-like structure if present
    if hasattr(page, "get") or isinstance(page, dict):
        polys = page.get("rec_polys") or page.get("dt_polys") or []
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or [1.0] * len(texts)
        boxes = []
        for i, poly in enumerate(polys):
            text = texts[i] if i < len(texts) else ""
            score = float(scores[i]) if i < len(scores) else 1.0
            pts = np.array(poly, dtype="float32").reshape(-1, 2).tolist()
            boxes.append([pts, (text, score)])
        return boxes

    return []


def normalize_ocr_text(text: str) -> str:
    """Dynamic, dictionary-free post-processing for OCR text.

    Applies universal typographical and punctuation heuristics without
    relying on static English word dictionaries:
    1. Removes stray pen-lift dots trapped inside words (e.g. 'dire.ctly' -> 'directly').
    2. Inserts missing spaces after punctuation (e.g. 'sustainability.Cross-docking' -> 'sustainability. Cross-docking', 'times.Walmart' -> 'times. Walmart').
    3. Cleans unspaced apostrophes/quotes between words (e.g. "Gigaton'to" -> "Gigaton to").
    4. Dynamically corrects common cursive Latinate adjective suffix ligatures (-tire -> -tive, -sire -> -sive).
    """
    if not text:
        return ""

    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue

        # 1. Remove stray dots trapped inside words (e.g. "dire.ctly" -> "directly")
        line = re.sub(r"(?<=[a-zA-Z])\.(?=[a-zA-Z])", "", line)

        # 2. Add missing space after punctuation when directly followed by a letter or number
        line = re.sub(r"([.,;:!?])(?=[a-zA-Z0-9])", r"\1 ", line)

        # 3. Clean unspaced apostrophe or quote between words (e.g. "Gigaton'to" -> "Gigaton to")
        line = re.sub(r"(?<=[a-zA-Z])['`\"](?=[a-zA-Z]{2,})", " ", line)

        # 4. Dynamic cursive suffix heuristic: in cursive handwriting, the letter 'r' frequently
        # mimics 'v' in common Latinate adjective suffixes (-tive, -sive mistaken as -tire, -sire)
        line = re.sub(r"\b([a-zA-Z]{3,})(?:tire)\b", r"\1tive", line)
        line = re.sub(r"\b([a-zA-Z]{3,})(?:sire)\b", r"\1sive", line)

        # 5. Clean up any duplicated whitespace
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)

    return "\n".join(lines).strip()


def process_handwritten_note(
    image_bytes: bytes,
    engine: Optional[str] = None,
) -> Dict[str, Any]:
    """Complete handwriting transcription pipeline:

    1. Decode & evaluate sharpness.
    2. Detect line bounding boxes via PaddleOCR.
    3. Transcribe lines using chosen engine ('paddle' default or 'trocr').
    """
    engine_used = (engine or DEFAULT_OCR_ENGINE).lower().strip()
    if engine_used not in ("paddle", "trocr"):
        engine_used = "paddle"

    img_cv = decode_image_bytes(image_bytes)
    empty_result = {
        "scan_quality": "poor",
        "handwriting_detected": 0.0,
        "retake": True,
        "detected_text": "",
        "boxes": [],
        "engine": engine_used,
    }
    if img_cv is None or img_cv.size == 0:
        return empty_result

    t_det_start = time.perf_counter()
    logger.info("[Step 1/2 Text Detection] Running PaddleOCR to locate handwritten lines...")
    scan_quality, _ = evaluate_scan_quality(img_cv)
    boxes = detect_text_boxes(img_cv)
    t_det_duration = time.perf_counter() - t_det_start
    logger.info(f"[Step 1/2 Text Detection] Found {len(boxes)} text box(es) in {t_det_duration:.2f}s")

    if not boxes:
        return {
            "scan_quality": scan_quality,
            "handwriting_detected": 0.0,
            "retake": True,
            "detected_text": "",
            "boxes": [],
            "engine": engine_used,
        }

    # Sort boxes top-to-bottom for natural reading order
    try:
        boxes = sorted(boxes, key=lambda b: min(pt[1] for pt in b[0]))
    except Exception:
        pass

    generated_text = ""

    if engine_used == "paddle":
        logger.info("[Step 2/2 Text Recognition] Using PaddleOCR CTC visual recognition engine...")
        t_paddle_rec = time.perf_counter()
        recognized_lines: List[str] = []
        for box_info in boxes:
            if len(box_info) > 1 and box_info[1] and len(box_info[1]) > 0:
                line_text = str(box_info[1][0]).strip()
                if line_text:
                    recognized_lines.append(line_text)
        generated_text = "\n".join(recognized_lines).strip()
        elapsed_rec = time.perf_counter() - t_paddle_rec
        logger.info(
            f"[Step 2/2 Text Recognition] Extracted {len(recognized_lines)} line(s) "
            f"via PaddleOCR CTC in {elapsed_rec:.2f}s."
        )

    elif engine_used == "trocr":
        processor, model = get_trocr_models()
        if processor is not None and model is not None:
            pil_images: List[Image.Image] = []
            for box_info in boxes:
                try:
                    pts = np.array(box_info[0], dtype="float32")
                    warped = four_point_transform(img_cv, pts)
                    if warped.shape[0] > 4 and warped.shape[1] > 4:
                        pil_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                        pil_images.append(pil_img)
                except Exception as exc:
                    logger.debug(f"Skipping malformed box during warp: {exc}")
                    continue

            total_lines = len(pil_images)
            if total_lines > 0:
                logger.info(f"[Step 2/2 Text Recognition] Transcribing {total_lines} line(s) with TrOCR...")
                decoded_lines: List[str] = []
                t_ocr_start = time.perf_counter()

                # Process in mini-batches of 2 for CPU memory efficiency and live progress feedback
                BATCH_SIZE = 2
                for batch_idx in range(0, total_lines, BATCH_SIZE):
                    batch_slice = pil_images[batch_idx : batch_idx + BATCH_SIZE]
                    t_batch_start = time.perf_counter()
                    try:
                        import torch
                        device = next(model.parameters()).device
                        pixel_values = processor(images=batch_slice, return_tensors="pt").pixel_values.to(device)
                        with torch.no_grad():
                            generated_ids = model.generate(
                                pixel_values,
                                max_new_tokens=85,
                                early_stopping=True,
                                no_repeat_ngram_size=3,
                            )
                        batch_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                        decoded_lines.extend(batch_texts)

                        batch_duration = time.perf_counter() - t_batch_start
                        done_count = min(batch_idx + len(batch_slice), total_lines)
                        pct = int((done_count / total_lines) * 100)
                        snippet = " | ".join(t.strip() for t in batch_texts if t.strip())[:60]
                        logger.info(
                            f"  -> Progress: {done_count}/{total_lines} lines ({pct}%) "
                            f"in {batch_duration:.2f}s | \"{snippet}\""
                        )
                    except Exception as exc:
                        logger.error(
                            f"TrOCR text recognition error on lines {batch_idx + 1}-{batch_idx + len(batch_slice)}: {exc}"
                        )

                total_ocr_time = time.perf_counter() - t_ocr_start
                logger.info(f"[Step 2/2 Text Recognition] Completed {total_lines} line(s) in {total_ocr_time:.2f}s")
                generated_text = "\n".join(t for t in decoded_lines if t.strip()).strip()

        # Fall back to PaddleOCR text if TrOCR produced no lines
        if not generated_text:
            fallback_lines = [
                str(b[1][0]).strip()
                for b in boxes
                if len(b) > 1 and b[1] and len(b[1]) > 0 and str(b[1][0]).strip()
            ]
            if fallback_lines:
                logger.warning("TrOCR yielded no output; falling back to PaddleOCR text.")
                generated_text = "\n".join(fallback_lines)

    # Dynamic post-processing: punctuation spacing, stray dot removal, and ligature normalization
    if generated_text:
        raw_char_count = len(generated_text)
        generated_text = normalize_ocr_text(generated_text)
        logger.info(
            f"[Post-Processing] Dynamically normalized OCR text ({raw_char_count} -> {len(generated_text)} chars)."
        )

    # Calculate handwriting detection confidence score
    if len(generated_text) > 5:
        handwriting_detected = min(95.0 + (len(generated_text) * 0.1), 99.9)
    else:
        # Fallback character estimation from box count
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

    return {
        "scan_quality": scan_quality,
        "handwriting_detected": float(round(handwriting_detected, 1)),
        "retake": retake,
        "detected_text": generated_text,
        "boxes": boxes,
        "engine": engine_used,
    }

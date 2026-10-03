"""Computer vision image preprocessing, blur detection, and geometry extraction."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from central_ai.core.logging import logger

# Domain Threshold Constants
BLUR_VARIANCE_THRESHOLD = 50.0
HIGH_PRESSURE_INK_DENSITY = 0.08
LOW_PRESSURE_INK_DENSITY = 0.02
WIDE_SPACING_PIXELS = 35.0
TIGHT_SPACING_PIXELS = 15.0
LARGE_WRITING_HEIGHT = 60.0
SMALL_WRITING_HEIGHT = 20.0


def decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode raw image bytes to an OpenCV BGR numpy array safely."""
    if not image_bytes:
        return None
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        logger.warning(f"Failed to decode image bytes: {exc}")
        return None


def evaluate_scan_quality(img_cv: np.ndarray) -> Tuple[str, float]:
    """Evaluate image sharpness using Laplacian variance.

    Returns
    -------
    (scan_quality, variance) : ("good" | "poor", float)
    """
    if img_cv is None or img_cv.size == 0:
        return "poor", 0.0

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    scan_quality = "good" if variance_of_laplacian > BLUR_VARIANCE_THRESHOLD else "poor"
    return scan_quality, variance_of_laplacian


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 polygon points consistently: top-left, top-right, bottom-right, bottom-left."""
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    right_most = right_most[np.argsort(right_most[:, 1]), :]
    (tr, br) = right_most

    return np.array([tl, tr, br, bl], dtype="float32")


def expand_polygon(
    pts: np.ndarray,
    pad_x_ratio: float = 0.04,
    pad_y_ratio: float = 0.18,
    img_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Expand 4-point polygon outward by pad_x_ratio and pad_y_ratio to prevent cursive clipping."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width = max(float(np.linalg.norm(tr - tl)), float(np.linalg.norm(br - bl)), 1.0)
    height = max(float(np.linalg.norm(bl - tl)), float(np.linalg.norm(br - tr)), 1.0)

    dx = width * pad_x_ratio
    dy = height * pad_y_ratio

    expanded = np.array([
        [tl[0] - dx, tl[1] - dy],
        [tr[0] + dx, tr[1] - dy],
        [br[0] + dx, br[1] + dy],
        [bl[0] - dx, bl[1] + dy],
    ], dtype="float32")

    if img_shape is not None:
        h_max, w_max = img_shape[:2]
        expanded[:, 0] = np.clip(expanded[:, 0], 0, w_max - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, h_max - 1)

    return expanded


def four_point_transform(
    image: np.ndarray,
    pts: np.ndarray,
    pad_x: float = 0.04,
    pad_y: float = 0.18,
) -> np.ndarray:
    """Perform a perspective warp to obtain a straightened rectangular crop with safety padding."""
    rect = expand_polygon(pts, pad_x_ratio=pad_x, pad_y_ratio=pad_y, img_shape=image.shape)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b), 1)

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b), 1)

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, transform_matrix, (max_width, max_height))


def extract_opencv_fallback_boxes(img_cv: np.ndarray) -> List[Any]:
    """Lightweight adaptive-threshold fallback when neural OCR is unavailable."""
    if img_cv is None or img_cv.size == 0:
        return []

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 12 or h < 12:
            continue
        pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        approx_chars = max(1, (w * h) // 400)
        boxes.append([pts, ("x" * min(approx_chars, 40), 0.5)])
    return boxes


def perform_advanced_analysis(image_bytes: bytes) -> Dict[str, str]:
    """Analyze handwriting stroke pressure, slant angle, letter spacing, and consistency."""
    img_cv = decode_image_bytes(image_bytes)
    fallback = {
        "slant": "Unable to analyze",
        "spacing": "Unable to analyze",
        "shapes": "Unable to analyze",
        "relative_size": "Unable to analyze",
        "consistency": "Unable to analyze",
        "stroke_geometry": "Unable to analyze",
    }
    if img_cv is None or img_cv.size == 0:
        return fallback

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # 1. Ink density estimation (Stroke Pressure)
    ink_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    density = ink_pixels / float(total_pixels) if total_pixels > 0 else 0.0

    if density > HIGH_PRESSURE_INK_DENSITY:
        stroke_geometry = "Heavy, thick strokes (High pressure)"
    elif density < LOW_PRESSURE_INK_DENSITY:
        stroke_geometry = "Light, thin strokes (Low pressure)"
    else:
        stroke_geometry = "Moderate, balanced strokes (Medium pressure)"

    # 2. Contour extraction for geometric metrics
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles: List[float] = []
    aspect_ratios: List[float] = []
    heights: List[float] = []
    x_positions: List[int] = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            heights.append(h)
            x_positions.append(x)
            aspect_ratios.append(w / float(h))
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            angles.append(angle)

    # Slant evaluation
    if angles:
        avg_angle = float(np.mean(angles))
        if avg_angle > 10.0:
            slant = f"Forward leaning ({avg_angle:.1f} deg)"
        elif avg_angle < -10.0:
            slant = f"Backward leaning ({avg_angle:.1f} deg)"
        else:
            slant = f"Vertical / Upright ({avg_angle:.1f} deg)"
    else:
        slant = "Vertical / Upright (default)"

    # Letter shape & loop characteristics
    if aspect_ratios:
        avg_aspect_ratio = float(np.mean(aspect_ratios))
        if avg_aspect_ratio > 1.2:
            shapes = "Wide, open curves and broad letter spacing"
        elif avg_aspect_ratio < 0.8:
            shapes = "Narrow, compressed loops and sharp angles"
        else:
            shapes = "Rounded, balanced loop structures"
    else:
        shapes = "Rounded, balanced loop structures"

    # Spacing analysis
    if len(x_positions) > 2:
        x_positions.sort()
        gaps = np.diff(x_positions)
        valid_gaps = [g for g in gaps if 5 < g < 100]
        if valid_gaps:
            avg_gap = float(np.mean(valid_gaps))
            if avg_gap > WIDE_SPACING_PIXELS:
                spacing = "Wide word and letter spacing"
            elif avg_gap < TIGHT_SPACING_PIXELS:
                spacing = "Tight, condensed word spacing"
            else:
                spacing = "Moderate, even spacing"
        else:
            spacing = "Moderate, even spacing"
    else:
        spacing = "Moderate, even spacing"

    # Relative size classification
    if heights:
        avg_height = float(np.mean(heights))
        if avg_height > LARGE_WRITING_HEIGHT:
            relative_size = "Large, expansive writing"
        elif avg_height < SMALL_WRITING_HEIGHT:
            relative_size = "Small, precise writing"
        else:
            relative_size = "Medium proportional size"
    else:
        relative_size = "Medium proportional size"

    # Uniformity & consistency
    h_variance = float(np.std(heights)) if heights else 15.0
    a_variance = float(np.std(angles)) if angles else 20.0
    if heights and angles:
        if h_variance < 10.0 and a_variance < 15.0:
            consistency = "Highly uniform and methodical"
        elif h_variance > 20.0 or a_variance > 30.0:
            consistency = "Variable, dynamic and expressive"
        else:
            consistency = "Generally consistent with natural variation"
    else:
        consistency = "Generally consistent"

    # Normalized physical metrics for ML training and continuous behavioral scoring
    metrics: Dict[str, float] = {
        "slant_angle": float(np.mean(angles)) if angles else 0.0,
        "slant_std": a_variance,
        "pressure_density": float(density),
        "aspect_ratio_mean": float(np.mean(aspect_ratios)) if aspect_ratios else 1.0,
        "letter_height_mean": float(np.mean(heights)) if heights else 30.0,
        "letter_height_std": h_variance,
        "word_gap_mean": float(np.mean(valid_gaps)) if (len(x_positions) > 2 and valid_gaps) else 25.0,
        "word_gap_std": float(np.std(valid_gaps)) if (len(x_positions) > 2 and valid_gaps) else 10.0,
        "uniformity_score": float(np.clip(1.0 - ((h_variance / 40.0) + (a_variance / 60.0)) / 2.0, 0.0, 1.0)),
    }

    return {
        "slant": slant,
        "spacing": spacing,
        "shapes": shapes,
        "relative_size": relative_size,
        "consistency": consistency,
        "stroke_geometry": stroke_geometry,
        "metrics": metrics,
    }

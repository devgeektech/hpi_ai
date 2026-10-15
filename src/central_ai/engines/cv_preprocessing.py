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

# Image Quality Gate Thresholds (7-point HPI Specification)
BLUR_VARIANCE_FAILED = 50.0
BLUR_VARIANCE_WARNING = 85.0
GLARE_RATIO_FAILED = 0.18
GLARE_RATIO_WARNING = 0.08
CROPPING_TOUCH_FAILED = 0.20
CROPPING_TOUCH_WARNING = 0.08
SKEW_ANGLE_FAILED = 18.0
SKEW_ANGLE_WARNING = 8.0
MIN_INK_DENSITY_VISIBLE = 0.005
MIN_IMAGE_DIMENSION = 300
INCOMPLETE_PAGE_ASPECT_FAILED = 0.20
INCOMPLETE_PAGE_ASPECT_WARNING = 0.30


class QualityReport(tuple):
    """Quality gating report that subclasses tuple (scan_quality_str, variance_float)
    for 100% backward compatibility with legacy unpacking:
        scan_quality, variance = evaluate_scan_quality(img)

    While also providing complete 7-point HPI evaluation properties and serialization.
    """

    def __new__(
        cls,
        overall_status: str,
        variance: float,
        rescan_required: bool,
        failed_page: Optional[int],
        issues: List[Dict[str, Any]],
        metrics: Dict[str, float],
    ):
        legacy_status = "good" if overall_status in ("Good", "Warning") else "poor"
        instance = super().__new__(cls, (legacy_status, variance))
        instance.overall_status = overall_status
        instance.variance = variance
        instance.rescan_required = rescan_required
        instance.failed_page = failed_page
        instance.issues = issues
        instance.metrics = metrics
        return instance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "rescan_required": self.rescan_required,
            "failed_page": self.failed_page,
            "issues": self.issues,
            "metrics": self.metrics,
        }


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


def evaluate_scan_quality(
    img_cv_or_bytes: np.ndarray | bytes,
    page_number: int = 1,
) -> QualityReport:
    """Evaluate submitted handwriting image against the 7-point HPI Image Quality Gate:
    1. Blur (Laplacian variance)
    2. Glare (Overexposure / flash washout)
    3. Cropping (Margin stroke truncation)
    4. Perspective problems (Skew angle)
    5. Incomplete page (Aspect ratio / truncated page geometry)
    6. Whether handwriting is actually visible (Foreground ink density)
    7. Whether the page is usable for analysis (Resolution and image decode validity)

    Classifies image as 'Good', 'Warning', or 'Failed'.
    """
    if isinstance(img_cv_or_bytes, bytes):
        img_cv = decode_image_bytes(img_cv_or_bytes)
    else:
        img_cv = img_cv_or_bytes

    if img_cv is None or getattr(img_cv, "size", 0) == 0:
        return QualityReport(
            overall_status="Failed",
            variance=0.0,
            rescan_required=True,
            failed_page=page_number,
            issues=[
                {
                    "criterion": "usability",
                    "severity": "Failed",
                    "value": 0.0,
                    "threshold": 1.0,
                    "message": "Uploaded image could not be decoded. Please upload a valid JPG or PNG file.",
                }
            ],
            metrics={
                "blur_variance": 0.0,
                "glare_ratio": 0.0,
                "cropping_boundary_touch_ratio": 0.0,
                "skew_angle": 0.0,
                "ink_density": 0.0,
                "aspect_ratio": 0.0,
                "quality_confidence": 0.0,
            },
        )

    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    total_pixels = float(h * w)

    issues: List[Dict[str, Any]] = []

    # 1. Page Usability (Resolution & Dimension Check)
    if h < MIN_IMAGE_DIMENSION or w < MIN_IMAGE_DIMENSION:
        issues.append({
            "criterion": "usability",
            "severity": "Failed",
            "value": float(min(h, w)),
            "threshold": float(MIN_IMAGE_DIMENSION),
            "message": f"Image resolution ({w}x{h}) is too low to be usable for analysis. Minimum size is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}.",
        })

    # 2. Incomplete Page (Extreme aspect ratio check)
    aspect_ratio = min(w, h) / max(w, h)
    if aspect_ratio < INCOMPLETE_PAGE_ASPECT_FAILED:
        issues.append({
            "criterion": "incomplete_page",
            "severity": "Failed",
            "value": round(aspect_ratio, 3),
            "threshold": INCOMPLETE_PAGE_ASPECT_FAILED,
            "message": "Incomplete page detected. Image appears to be a cropped strip or torn fragment rather than a full page.",
        })
    elif aspect_ratio < INCOMPLETE_PAGE_ASPECT_WARNING:
        issues.append({
            "criterion": "incomplete_page",
            "severity": "Warning",
            "value": round(aspect_ratio, 3),
            "threshold": INCOMPLETE_PAGE_ASPECT_WARNING,
            "message": "Unusual page aspect ratio. Please ensure the full page is captured.",
        })

    # 3. Blur Evaluation (Laplacian variance)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_variance < BLUR_VARIANCE_FAILED:
        issues.append({
            "criterion": "blur",
            "severity": "Failed",
            "value": round(blur_variance, 2),
            "threshold": BLUR_VARIANCE_FAILED,
            "message": "Image is excessively blurry. Please hold camera steady in good lighting.",
        })
    elif blur_variance < BLUR_VARIANCE_WARNING:
        issues.append({
            "criterion": "blur",
            "severity": "Warning",
            "value": round(blur_variance, 2),
            "threshold": BLUR_VARIANCE_WARNING,
            "message": "Slight blur detected. Fine stroke details may have soft edges.",
        })

    # 4. Handwriting Visibility (Foreground ink extraction)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    ink_count = cv2.countNonZero(thresh)
    ink_density = ink_count / total_pixels if total_pixels > 0 else 0.0

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] > 10 and cv2.boundingRect(cnt)[3] > 10]

    if ink_density < MIN_INK_DENSITY_VISIBLE or len(text_contours) < 3:
        issues.append({
            "criterion": "visibility",
            "severity": "Failed",
            "value": round(ink_density, 4),
            "threshold": MIN_INK_DENSITY_VISIBLE,
            "message": "Handwriting is not visible on this page. Please submit a page with clear handwritten text.",
        })

    # 5. Glare / Overexposure evaluation
    overexposed_count = float(np.sum(gray >= 250))
    glare_ratio = overexposed_count / total_pixels if total_pixels > 0 else 0.0
    mean_luminance = float(gray.mean())

    if ink_density >= MIN_INK_DENSITY_VISIBLE:
        if mean_luminance <= 220.0:
            if glare_ratio > GLARE_RATIO_FAILED:
                issues.append({
                    "criterion": "glare",
                    "severity": "Failed",
                    "value": round(glare_ratio, 4),
                    "threshold": GLARE_RATIO_FAILED,
                    "message": f"Harsh glare or flash washout detected across {glare_ratio * 100:.1f}% of the page. Avoid direct flash or reflections.",
                })
            elif glare_ratio > GLARE_RATIO_WARNING:
                issues.append({
                    "criterion": "glare",
                    "severity": "Warning",
                    "value": round(glare_ratio, 4),
                    "threshold": GLARE_RATIO_WARNING,
                    "message": f"Moderate light glare detected across {glare_ratio * 100:.1f}% of the page.",
                })
        elif glare_ratio > 0.90:
            issues.append({
                "criterion": "glare",
                "severity": "Failed",
                "value": round(glare_ratio, 4),
                "threshold": 0.90,
                "message": "Severe overexposure glare detected across the page. Background is washed out.",
            })

    # 6. Cropping & Page boundary check
    margin_x = max(int(w * 0.02), 8)
    margin_y = max(int(h * 0.02), 8)
    boundary_touch_count = 0
    angles: List[float] = []

    for cnt in text_contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bx <= margin_x or by <= margin_y or (bx + bw) >= (w - margin_x) or (by + bh) >= (h - margin_y):
            boundary_touch_count += 1

        rect = cv2.minAreaRect(cnt)
        ang = rect[2]
        if ang < -45:
            ang += 90
        elif ang > 45:
            ang -= 90
        angles.append(ang)

    boundary_ratio = boundary_touch_count / float(len(text_contours)) if text_contours else 0.0
    if boundary_ratio > CROPPING_TOUCH_FAILED and len(text_contours) >= 5:
        issues.append({
            "criterion": "cropping",
            "severity": "Failed",
            "value": round(boundary_ratio, 3),
            "threshold": CROPPING_TOUCH_FAILED,
            "message": "Handwriting is cut off or cropped at the page margins. Please capture the complete page.",
        })
    elif boundary_ratio > CROPPING_TOUCH_WARNING and len(text_contours) >= 5:
        issues.append({
            "criterion": "cropping",
            "severity": "Warning",
            "value": round(boundary_ratio, 3),
            "threshold": CROPPING_TOUCH_WARNING,
            "message": "Several handwritten words lie very close to the outer edge of the image.",
        })

    # 7. Perspective distortion & Skew check
    avg_skew = float(np.abs(np.mean(angles))) if angles else 0.0
    if avg_skew > SKEW_ANGLE_FAILED:
        issues.append({
            "criterion": "perspective",
            "severity": "Failed",
            "value": round(avg_skew, 2),
            "threshold": SKEW_ANGLE_FAILED,
            "message": f"Extreme perspective angle detected ({avg_skew:.1f}°). Please take the photograph straight-on from above.",
        })
    elif avg_skew > SKEW_ANGLE_WARNING:
        issues.append({
            "criterion": "perspective",
            "severity": "Warning",
            "value": round(avg_skew, 2),
            "threshold": SKEW_ANGLE_WARNING,
            "message": f"Noticeable perspective skew detected ({avg_skew:.1f}°). Straight-on angle recommended.",
        })

    # Determine overall status: Good, Warning, Failed
    has_failed = any(issue["severity"] == "Failed" for issue in issues)
    has_warning = any(issue["severity"] == "Warning" for issue in issues)

    if has_failed:
        overall_status = "Failed"
        rescan_required = True
        failed_page = page_number
    elif has_warning:
        overall_status = "Warning"
        rescan_required = False
        failed_page = None
    else:
        overall_status = "Good"
        rescan_required = False
        failed_page = None

    # Calculate normalized quality confidence (0-100)
    blur_score = min(100.0, (blur_variance / 120.0) * 100.0)
    glare_score = max(0.0, (1.0 - (glare_ratio / 0.15)) * 100.0)
    crop_score = max(0.0, (1.0 - (boundary_ratio / 0.25)) * 100.0)
    skew_score = max(0.0, (1.0 - (avg_skew / 25.0)) * 100.0)
    quality_confidence = float(np.clip(
        0.35 * blur_score + 0.25 * glare_score + 0.20 * crop_score + 0.20 * skew_score,
        0.0,
        100.0,
    ))
    if has_failed:
        quality_confidence = min(quality_confidence, 35.0)

    metrics = {
        "blur_variance": round(blur_variance, 2),
        "glare_ratio": round(glare_ratio, 4),
        "cropping_boundary_touch_ratio": round(boundary_ratio, 4),
        "skew_angle": round(avg_skew, 2),
        "ink_density": round(ink_density, 4),
        "aspect_ratio": round(aspect_ratio, 3),
        "quality_confidence": round(quality_confidence, 2),
    }

    return QualityReport(
        overall_status=overall_status,
        variance=blur_variance,
        rescan_required=rescan_required,
        failed_page=failed_page,
        issues=issues,
        metrics=metrics,
    )


def evaluate_image_quality_gate(
    img_cv_or_bytes: np.ndarray | bytes,
    page_number: int = 1,
) -> Dict[str, Any]:
    """Helper alias returning a dict representation of the QualityReport."""
    return evaluate_scan_quality(img_cv_or_bytes, page_number=page_number).to_dict()


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

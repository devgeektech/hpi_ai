import os

# Windows OpenMP + Paddle safety
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

ocr_detector = None
trocr_processor = None
trocr_model = None


def load_paddleocr():
    global ocr_detector
    if ocr_detector is None:
        print("Loading PaddleOCR 2.x for text detection...")
        # Pinned stack: paddleocr==2.7.3 + paddlepaddle==2.6.2 (stable on Windows CPU)
        ocr_detector = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
            use_gpu=False,
            enable_mkldnn=False,
        )


def load_trocr():
    global trocr_processor, trocr_model
    if trocr_processor is None or trocr_model is None:
        print("Loading TrOCR for handwriting recognition...")
        trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        trocr_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trocr_model = trocr_model.to(device)
        print(f"TrOCR Models loaded successfully (device: {device}).")


def _boxes_from_opencv(img_cv):
    """Lightweight ink-region fallback when PaddleOCR fails on this machine."""
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


def _run_paddle_ocr(img_cv):
    """Run PaddleOCR 2.x and return legacy box list: [[points, (text, score)], ...]."""
    try:
        # PaddleOCR 2.x API
        raw = ocr_detector.ocr(img_cv, cls=True)
    except TypeError:
        # Safety if a 3.x build is still installed
        try:
            raw = ocr_detector.ocr(img_cv)
        except Exception as exc:
            print(f"PaddleOCR failed ({exc}); using OpenCV fallback boxes.")
            return _boxes_from_opencv(img_cv)
    except Exception as exc:
        print(f"PaddleOCR failed ({exc}); using OpenCV fallback boxes.")
        return _boxes_from_opencv(img_cv)

    if not raw or not raw[0]:
        return []

    page = raw[0]

    # PaddleOCR 2.x: list of [box, (text, score)]
    if isinstance(page, list):
        return page

    # PaddleOCR 3.x OCRResult dict-like (if somehow still present)
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


def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost
    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def analyze_image_quality(image_bytes: bytes) -> dict:
    load_paddleocr()

    np_img = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img_cv is None:
        return {"scan_quality": "poor", "handwriting_detected": 0.0, "retake": True}

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    scan_quality = "good" if variance_of_laplacian > 50 else "poor"

    boxes = _run_paddle_ocr(img_cv)

    if not boxes:
        return {"scan_quality": scan_quality, "handwriting_detected": 0.0, "retake": True}

    total_chars = sum(len(box[1][0]) for box in boxes if box[1] and box[1][0])

    if total_chars < 35:
        handwriting_detected = max(10.0, total_chars * 1.0)
    else:
        handwriting_detected = min(75.0 + (total_chars * 0.2), 99.9)

    retake = handwriting_detected < 50.0

    return {
        "scan_quality": scan_quality,
        "handwriting_detected": float(handwriting_detected),
        "retake": retake,
    }


def process_handwritten_note(image_bytes: bytes) -> dict:
    load_paddleocr()
    load_trocr()

    np_img = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img_cv is None:
        return {
            "scan_quality": "poor",
            "handwriting_detected": 0.0,
            "retake": True,
            "detected_text": "",
        }

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    scan_quality = "good" if variance_of_laplacian > 50 else "poor"

    boxes = _run_paddle_ocr(img_cv)

    if not boxes:
        return {
            "scan_quality": scan_quality,
            "handwriting_detected": 0.0,
            "retake": True,
            "detected_text": "",
        }

    boxes = sorted(boxes, key=lambda box: min(pt[1] for pt in box[0]))

    pil_images = []
    for box_info in boxes:
        pts = np.array(box_info[0], dtype="float32")
        warped = four_point_transform(img_cv, pts)
        pil_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        pil_images.append(pil_img)

    device = next(trocr_model.parameters()).device
    pixel_values = trocr_processor(images=pil_images, return_tensors="pt").pixel_values.to(
        device
    )
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values, max_new_tokens=50)
    detected_texts = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)

    generated_text = "\n".join(detected_texts).strip()

    print("\n" + "=" * 50)
    print("EXTRACTED HANDWRITTEN TEXT:")
    print("-" * 50)
    print(generated_text)
    print("=" * 50 + "\n")

    if len(generated_text) > 5:
        handwriting_detected = min(95.0 + (len(generated_text) * 0.1), 99.9)
    else:
        handwriting_detected = 0.0

    retake = handwriting_detected < 5.0

    return {
        "scan_quality": scan_quality,
        "handwriting_detected": float(handwriting_detected),
        "retake": retake,
        "detected_text": generated_text,
        "boxes": boxes,
    }


def perform_advanced_analysis(image_bytes: bytes) -> dict:
    np_img = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img_cv is None:
        return {
            "slant": "Unable to analyze",
            "spacing": "Unable to analyze",
            "shapes": "Unable to analyze",
            "relative_size": "Unable to analyze",
            "consistency": "Unable to analyze",
            "stroke_geometry": "Unable to analyze",
        }

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    ink_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    density = ink_pixels / float(total_pixels) if total_pixels > 0 else 0

    if density > 0.08:
        stroke_geometry = "Heavy, thick strokes (High pressure)"
    elif density < 0.02:
        stroke_geometry = "Light, thin strokes (Low pressure)"
    else:
        stroke_geometry = "Moderate, balanced strokes (Medium pressure)"

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    aspect_ratios = []
    heights = []
    x_coords = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            heights.append(h)
            x_coords.append(x)
            aspect_ratios.append(w / float(h))
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            angles.append(angle)

    if angles:
        avg_angle = np.mean(angles)
        if avg_angle > 10:
            slant = f"Forward leaning ({avg_angle:.1f} deg)"
        elif avg_angle < -10:
            slant = f"Backward leaning ({avg_angle:.1f} deg)"
        else:
            slant = f"Vertical / Upright ({avg_angle:.1f} deg)"
    else:
        slant = "Vertical / Upright (default)"

    if aspect_ratios:
        avg_ar = np.mean(aspect_ratios)
        if avg_ar > 1.2:
            shapes = "Wide, open curves and broad letter spacing"
        elif avg_ar < 0.8:
            shapes = "Narrow, compressed loops and sharp angles"
        else:
            shapes = "Rounded, balanced loop structures"
    else:
        shapes = "Rounded, balanced loop structures"

    if len(x_coords) > 2:
        x_coords.sort()
        gaps = np.diff(x_coords)
        valid_gaps = [g for g in gaps if 5 < g < 100]
        if valid_gaps:
            avg_gap = np.mean(valid_gaps)
            if avg_gap > 35:
                spacing = "Wide word and letter spacing"
            elif avg_gap < 15:
                spacing = "Tight, condensed word spacing"
            else:
                spacing = "Moderate, even spacing"
        else:
            spacing = "Moderate, even spacing"
    else:
        spacing = "Moderate, even spacing"

    if heights:
        avg_height = np.mean(heights)
        if avg_height > 60:
            relative_size = "Large, expansive writing"
        elif avg_height < 20:
            relative_size = "Small, precise writing"
        else:
            relative_size = "Medium proportional size"
    else:
        relative_size = "Medium proportional size"

    if heights and angles:
        h_var = np.std(heights)
        a_var = np.std(angles)
        if h_var < 10 and a_var < 15:
            consistency = "Highly uniform and methodical"
        elif h_var > 20 or a_var > 30:
            consistency = "Variable, dynamic and expressive"
        else:
            consistency = "Generally consistent with natural variation"
    else:
        consistency = "Generally consistent"

    return {
        "slant": slant,
        "spacing": spacing,
        "shapes": shapes,
        "relative_size": relative_size,
        "consistency": consistency,
        "stroke_geometry": stroke_geometry,
    }

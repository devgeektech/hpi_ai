"""AI and Computer Vision engines for handwriting processing."""

from .cv_preprocessing import (
    decode_image_bytes,
    evaluate_scan_quality,
    evaluate_image_quality_gate,
    QualityReport,
    perform_advanced_analysis,
    four_point_transform,
)
from .ocr_engine import (
    process_handwritten_note,
    detect_text_boxes,
)
from .active_learning import (
    record_and_train_sample,
    get_model_status,
)

__all__ = [
    "decode_image_bytes",
    "evaluate_scan_quality",
    "evaluate_image_quality_gate",
    "QualityReport",
    "perform_advanced_analysis",
    "four_point_transform",
    "process_handwritten_note",
    "detect_text_boxes",
    "record_and_train_sample",
    "get_model_status",
]

"""Strongly-typed Pydantic schemas for handwriting analysis and HPI assessment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ImageQualityResponse(BaseModel):
    """Result of image quality validation check."""
    scan_quality: str = Field(..., description="Overall scan quality assessment ('good' or 'poor')")
    handwriting_detected: float = Field(..., description="Handwriting confidence score [0.0 - 100.0]")
    retake: bool = Field(..., description="True if image quality or text is insufficient and retake is advised")


class AdvancedFeaturesResponse(BaseModel):
    """Raw computer vision handwriting geometry features."""
    slant: str = Field(..., description="Letter slant angle and direction description")
    spacing: str = Field(..., description="Inter-word and letter spacing pattern")
    shapes: str = Field(..., description="Loop structure and character shape description")
    relative_size: str = Field(..., description="Relative character sizing classification")
    consistency: str = Field(..., description="Uniformity and baseline consistency assessment")
    stroke_geometry: str = Field(..., description="Estimated stroke pressure and line weight")


class FeatureSet(AdvancedFeaturesResponse):
    """Extended feature set including scan quality and transcription."""
    detected_text: str = Field(default="", description="Transcribed handwritten text")
    scan_quality: str = Field(default="unknown", description="Scan quality assessment")
    handwriting_detected: float = Field(default=0.0, description="Handwriting confidence percentage")
    retake: bool = Field(default=False, description="Whether a retake is recommended")
    numeric: Dict[str, float] = Field(default_factory=dict, description="Numeric biometric measurements")


class ScoreDetail(BaseModel):
    """Single HPI trait score."""
    name: str = Field(..., description="Trait name (e.g., Execution, Focus, Logic)")
    value: int = Field(..., ge=0, le=100, description="Normalized trait score [0 - 100]")


class InsightDetail(BaseModel):
    """Actionable psychological insight for an HPI trait."""
    title: str = Field(..., description="Trait title")
    overall_interpretation: str = Field(..., description="Summary interpretation")
    at_your_best: str = Field(..., description="Positive strength manifestation")
    what_to_watch: str = Field(..., description="Potential blind spots or risks")
    action_to_practise: str = Field(..., description="Practical actionable developmental habit")


class QualityIssue(BaseModel):
    """Specific image quality defect identified during gating."""
    criterion: str = Field(..., description="'blur' | 'glare' | 'cropping' | 'perspective' | 'incomplete_page' | 'visibility' | 'usability'")
    severity: str = Field(..., description="'Warning' | 'Failed'")
    value: float = Field(..., description="Measured defect value")
    threshold: float = Field(..., description="Violated benchmark threshold")
    message: str = Field(..., description="Human-readable remediation instructions")


class QualityGateResult(BaseModel):
    """Complete 7-point image quality gating evaluation."""
    overall_status: str = Field(..., description="'Good' | 'Warning' | 'Failed'")
    rescan_required: bool = Field(..., description="True if analysis must abort and user rescan")
    failed_page: Optional[int] = Field(default=None, description="Specific page number needing rescan")
    issues: List[QualityIssue] = Field(default_factory=list, description="List of quality issues identified")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Raw quantitative metrics")


class AssessmentProfile(BaseModel):
    """Complete HPI assessment profile."""
    standout_strength: str = Field(..., description="Top standout strength title")
    scores: List[ScoreDetail] = Field(default_factory=list, description="Ranked trait scores")
    insights: List[InsightDetail] = Field(default_factory=list, description="Accompanying insights")


class OcrLineItem(BaseModel):
    """Single line extracted via OCR."""
    line_number: int = Field(..., description="1-indexed line order")
    text: str = Field(..., description="Recognized line transcription")
    confidence: float = Field(..., description="Line confidence percentage [0.0 - 100.0]")
    bounding_box: Optional[List[List[float]]] = Field(default=None, description="4-point polygon coordinates")


class OcrExtractionResult(BaseModel):
    """Extracted text recognition results."""
    detected_text: str = Field(default="", description="Full handwritten text transcription")
    total_lines: int = Field(default=0, description="Total detected lines")
    total_words: int = Field(default=0, description="Total word count")
    total_characters: int = Field(default=0, description="Total character count")
    ocr_confidence: float = Field(default=0.0, description="Overall OCR confidence percentage")
    engine: str = Field(default="paddle", description="OCR engine utilized")
    lines: List[OcrLineItem] = Field(default_factory=list, description="Line-by-line transcription details")


class HandwritingFeaturesResult(BaseModel):
    """Objective computer vision handwriting measurements (zero psychological interpretation)."""
    slant: str = Field(default="Unknown", description="Slant direction description")
    spacing: str = Field(default="Unknown", description="Spacing pattern description")
    shapes: str = Field(default="Unknown", description="Loop and character shape geometry")
    relative_size: str = Field(default="Unknown", description="Proportional size classification")
    consistency: str = Field(default="Unknown", description="Baseline uniformity classification")
    stroke_geometry: str = Field(default="Unknown", description="Stroke pressure and line weight")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Quantitative biometric metrics")


class ImageDimension(BaseModel):
    width: int = Field(default=0, description="Image width in pixels")
    height: int = Field(default=0, description="Image height in pixels")


class AnalysisMetadata(BaseModel):
    filename: str = Field(..., description="Uploaded image filename")
    image_size_bytes: int = Field(..., description="File size in bytes")
    dimensions: ImageDimension = Field(default_factory=ImageDimension, description="Image pixel dimensions")
    project_id: str = Field(..., description="Project / tenant identifier")
    analyzed_at: str = Field(..., description="ISO 8601 UTC timestamp")
    processing_time_seconds: float = Field(default=0.0, description="Elapsed processing time in seconds")


class AnalysisResultPayload(BaseModel):
    """Handwriting perception response containing image quality gating, text extraction, and biometrics."""
    result_id: str = Field(..., description="Unique result tracking ID")
    status: str = Field(..., description="Processing status ('completed', 'warning', or 'failed_quality_check')")
    message: str = Field(..., description="Status summary and remediation instructions")
    quality_gate: QualityGateResult = Field(..., description="7-point image quality gate evaluation")
    ocr: OcrExtractionResult = Field(..., description="Extracted text and line-by-line transcription")
    handwriting_features: HandwritingFeaturesResult = Field(..., description="Objective physical handwriting stroke measurements")
    metadata: AnalysisMetadata = Field(..., description="Analysis execution metadata")


class ModelStatusResponse(BaseModel):
    """Current state and sample count of active learning model."""
    mode: str = Field(..., description="Current inference mode ('trained_supervised_ml' or 'calibrated_graphological_rules')")
    total_samples_trained: int = Field(..., description="Total verified handwriting samples learned")
    last_updated: Optional[str] = Field(default=None, description="ISO timestamp of last training iteration")
    model_version: str = Field(..., description="Model architecture or checkpoint version")


class TrainingResultResponse(BaseModel):
    """Response returned when submitting a training sample."""
    status: str = Field(..., description="'trained' or 'skipped'")
    sample_id: str = Field(..., description="Identifier of the recorded sample")
    total_samples_trained: int = Field(..., description="Total samples in dataset after this update")
    updated_at: Optional[str] = Field(default=None, description="Timestamp model was updated")
    recorded_scores: Dict[str, float] = Field(default_factory=dict, description="Verified scores associated with sample")
    metrics_summary: Dict[str, float] = Field(default_factory=dict, description="Biometric feature vector extracted from upload")


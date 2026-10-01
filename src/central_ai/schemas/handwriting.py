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


class AssessmentProfile(BaseModel):
    """Complete HPI assessment profile."""
    standout_strength: str = Field(..., description="Top standout strength title")
    scores: List[ScoreDetail] = Field(default_factory=list, description="Ranked trait scores")
    insights: List[InsightDetail] = Field(default_factory=list, description="Accompanying insights")


class AnalysisResultPayload(BaseModel):
    """Full handwriting analysis response payload."""
    result_id: str = Field(..., description="Unique result tracking ID")
    status: str = Field(..., description="Processing status ('completed' or 'low_confidence')")
    confidence: float = Field(..., description="Overall confidence percentage")
    project_id: str = Field(..., description="Originating project/tenant ID")
    org_id: str = Field(..., description="Originating organization ID")
    features: FeatureSet = Field(..., description="Extracted handwriting features")
    profile: AssessmentProfile = Field(..., description="Calculated HPI profile")
    model_version: str = Field(..., description="Model version tag")
    methodology_version: str = Field(..., description="Methodology version tag")
    rule_version: str = Field(..., description="Rule engine version tag")
    feature_schema_version: str = Field(..., description="Feature schema version tag")
    analyzed_at: str = Field(..., description="ISO 8601 UTC timestamp")
    upload_path: Optional[str] = Field(default=None, description="Optional local upload file path")

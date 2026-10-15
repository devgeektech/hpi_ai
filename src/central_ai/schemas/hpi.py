"""Pydantic schemas for HPI Individual Methodology, Findings, Confidence Vector, and Frameworks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ConfidenceVector(BaseModel):
    """6-Stage Multi-Dimensional Confidence Vector (Section 8)."""
    quality_confidence: float = Field(..., ge=0.0, le=100.0, description="Image sharpness, lighting, and margin quality")
    detection_confidence: float = Field(..., ge=0.0, le=100.0, description="Handwriting ink visibility confirmation")
    ocr_confidence: float = Field(..., ge=0.0, le=100.0, description="Text recognition and character extraction accuracy")
    feature_confidence: float = Field(..., ge=0.0, le=100.0, description="Contour and stroke measurement stability")
    methodology_confidence: float = Field(..., ge=0.0, le=100.0, description="Rule condition coverage across approved catalog")
    overall_confidence: float = Field(..., ge=0.0, le=100.0, description="Weighted composite determining review routing")


class HPIFinding(BaseModel):
    """Structured HPI Finding supported by measurable evidence and approved rule citation (Section 7)."""
    trait_name: str = Field(..., description="Approved HPI trait name (e.g., Execution, Focus, Logic)")
    score: int = Field(..., ge=0, le=100, description="Dynamically calculated trait score [0-100]")
    strength_title: str = Field(..., description="Actionable strength manifestation title")
    interpretation: str = Field(..., description="Behavioral interpretation narrative")
    evidence: str = Field(..., description="Measurable biometric evidence citation")
    rule_ref_id: str = Field(..., description="Approved methodology rule identifier (e.g. RULE-EXE-01)")
    confidence: float = Field(..., ge=0.0, le=100.0, description="Trait-level finding confidence percentage")
    rule_version: str = Field(..., description="Rule catalog version used")
    methodology_version: str = Field(..., description="Methodology specification version")
    review_status: str = Field(default="auto_approved", description="Review status of this finding ('auto_approved' | 'in_human_review' | 'reviewer_approved')")


class PRIMEDimension(BaseModel):
    """Single pillar of the PRIME™ framework."""
    name: str = Field(..., description="'Perception' | 'Response' | 'Interaction' | 'Momentum' | 'Evolution'")
    score: int = Field(..., ge=0, le=100, description="Pillar score [0-100]")
    descriptor: str = Field(..., description="Strategic behavioral orientation")
    evidence_citation: str = Field(..., description="Citation of findings supporting this pillar")


class PRIMEScorecard(BaseModel):
    """Complete 5-Pillar PRIME™ behavioral framework (Section 11)."""
    perception: PRIMEDimension
    response: PRIMEDimension
    interaction: PRIMEDimension
    momentum: PRIMEDimension
    evolution: PRIMEDimension
    prime_index: float = Field(..., ge=0.0, le=100.0, description="Overall PRIME composite index")


class ALIGNDimension(BaseModel):
    """Single dimension of the ALIGN™ action framework."""
    dimension: str = Field(..., description="'Awareness' | 'Leverage' | 'Interrupt' | 'Grow' | 'Next Moves'")
    focus_trait: str = Field(..., description="Primary HPI trait driving this action")
    recommendation: str = Field(..., description="Targeted behavioral recommendation")
    action_trigger: str = Field(..., description="Situational cue when to practice this recommendation")


class ALIGNRecommendations(BaseModel):
    """Complete 5-Dimension ALIGN™ developmental action plan (Section 12)."""
    awareness: ALIGNDimension
    leverage: ALIGNDimension
    interrupt: ALIGNDimension
    grow: ALIGNDimension
    next_moves: ALIGNDimension


class HPIAssessmentResult(BaseModel):
    """Complete approved HPI assessment profile output from Tier 2 reasoning layer."""
    result_id: str = Field(..., description="Unique assessment tracking ID")
    status: str = Field(..., description="'completed' | 'failed_quality_check' | 'warning'")
    review_status: str = Field(..., description="'auto_approved' | 'in_human_review' | 'reviewer_approved'")
    rescan_required: bool = Field(default=False, description="True if quality failed and rescan is required")
    standout_strength: str = Field(..., description="Top standout behavioral strength")
    findings: List[HPIFinding] = Field(default_factory=list, description="Approved evidence-backed HPI findings")
    prime: Optional[PRIMEScorecard] = Field(default=None, description="5-Pillar PRIME™ scorecard")
    align: Optional[ALIGNRecommendations] = Field(default=None, description="5-Dimension ALIGN™ action recommendations")
    confidence_vector: ConfidenceVector = Field(..., description="6-stage multi-dimensional confidence breakdown")
    methodology_version: str = Field(..., description="Methodology specification version")
    rule_version: str = Field(..., description="Rule catalog version")
    analyzed_at: str = Field(..., description="ISO 8601 UTC timestamp")
    project_id: str = Field(default="hpi", description="Originating project/tenant ID")
    message: Optional[str] = Field(default=None, description="Informational message or review routing notes")


class HPIAssessRequest(BaseModel):
    """Direct assessment request payload containing previously extracted physical biometrics.
    
    Can accept either:
    1. The raw physical 'metrics' dictionary
    2. The full perception response from POST /api/v1/handwriting/analyze directly!
    """
    result_id: Optional[str] = Field(default=None, description="Result ID from previous handwriting analysis")
    sample_id: Optional[str] = Field(default=None, description="Optional sample tracking identifier")
    metrics: Optional[Dict[str, float]] = Field(default=None, description="Quantitative physical metrics from CV extraction")
    handwriting_features: Optional[Dict[str, Any]] = Field(default=None, description="Handwriting features block from handwriting/analyze")
    quality_gate: Optional[Dict[str, Any]] = Field(default=None, description="Quality gate block from handwriting/analyze")
    ocr: Optional[Dict[str, Any]] = Field(default=None, description="OCR text block from handwriting/analyze")
    ocr_confidence: Optional[float] = Field(default=None, description="OCR text recognition confidence")
    quality_confidence: Optional[float] = Field(default=None, description="Image quality score")

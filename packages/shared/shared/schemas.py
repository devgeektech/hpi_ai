from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | processing | needs_review | completed | failed
    result_id: Optional[str] = None
    message: Optional[str] = None


class FeatureSet(BaseModel):
    slant: str
    spacing: str
    shapes: str
    relative_size: str
    consistency: str
    stroke_geometry: str
    detected_text: str = ""
    scan_quality: str = "unknown"
    handwriting_detected: float = 0.0
    numeric: Dict[str, float] = Field(default_factory=dict)


class ScoreDetail(BaseModel):
    name: str
    value: int


class InsightDetail(BaseModel):
    title: str
    overall_interpretation: str
    at_your_best: str
    what_to_watch: str
    action_to_practise: str


class AnalysisResultPayload(BaseModel):
    result_id: str
    job_id: str
    status: str
    features: Optional[FeatureSet] = None
    standout_strength: Optional[str] = None
    scores: List[ScoreDetail] = Field(default_factory=list)
    insights: List[InsightDetail] = Field(default_factory=list)
    confidence: float = 0.0
    model_version: str
    methodology_version: str
    rule_version: str
    feature_schema_version: str
    report_version: str = "report-v1"
    raw: Dict[str, Any] = Field(default_factory=dict)

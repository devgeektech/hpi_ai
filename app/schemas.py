from pydantic import BaseModel
from typing import List

class AnalyzeResponse(BaseModel):
    scan_quality: str
    handwriting_detected: float
    retake: bool

class InsightDetail(BaseModel):
    title: str
    overall_interpretation: str
    at_your_best: str
    what_to_watch: str
    action_to_practise: str

class ScoreDetail(BaseModel):
    name: str
    value: int

class ProfileResult(BaseModel):
    standout_strength: str
    scores: List[ScoreDetail]
    insights: List[InsightDetail]

class UploadResponse(BaseModel):
    scan_quality: str
    handwriting_detected: float
    retake: bool
    detected_text: str
    profile: ProfileResult

class JourneyWeek(BaseModel):
    week_number: int
    title: str
    description: str
    status: str  # e.g., "completed", "active", "locked"

class JourneyResponse(BaseModel):
    current_focus: str
    weeks_complete: int
    total_weeks: int
    weeks: List[JourneyWeek]

class AdvancedAnalysisResponse(BaseModel):
    slant: str
    spacing: str
    shapes: str
    relative_size: str
    consistency: str
    stroke_geometry: str

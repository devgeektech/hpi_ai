"""HPI Assessment Profile Generator.

Centralized engine mapping handwriting features and OCR text to HPI traits.
Uses explainable graphological heuristics and deterministic calibration.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional
import numpy as np

from central_ai.core.logging import logger

ALL_TRAITS: List[Dict[str, str]] = [
    {
        "name": "Communication",
        "strength_title": "Clear Expression",
        "overall": "You tend to communicate with clarity and intention, especially when you have time to structure your thoughts.",
        "best": "Clear, structured and thoughtful communication that inspires trust.",
        "watch": "Over-structuring can sometimes slow spontaneous conversations.",
        "action": "Ask one open-ended question before offering your solution.",
    },
    {
        "name": "Decision Style",
        "strength_title": "Strategic Clarity",
        "overall": "You tend to evaluate options with a long-term perspective.",
        "best": "Strategic, visionary, and forward-looking.",
        "watch": "Can overthink simple decisions.",
        "action": "Trust your initial gut feeling on smaller choices.",
    },
    {
        "name": "Adaptability",
        "strength_title": "Adaptive Agility",
        "overall": "You adjust quickly to new information and changing environments.",
        "best": "Flexible, resilient, and open-minded.",
        "watch": "Can lose focus on long-term goals.",
        "action": "Set one rigid milestone per week to anchor your flexibility.",
    },
    {
        "name": "Execution",
        "strength_title": "Structured Execution",
        "overall": "You drive tasks to completion with intense focus.",
        "best": "Reliable, consistent, and highly organized.",
        "watch": "May struggle when plans change abruptly.",
        "action": "Build buffer time into your daily schedule.",
    },
    {
        "name": "Creativity",
        "strength_title": "Innovative Thinking",
        "overall": "You approach problems from unique angles and value originality.",
        "best": "Imaginative, resourceful, and capable of connecting dots others miss.",
        "watch": "Can get lost in ideas without anchoring them to reality.",
        "action": "Jot down the practical steps required to test your best idea today.",
    },
    {
        "name": "Empathy",
        "strength_title": "Deep Intuition",
        "overall": "You naturally sense the emotional undertones of a situation.",
        "best": "Compassionate, supportive, and a great listener.",
        "watch": "Can absorb others' stress and become overwhelmed.",
        "action": "Take 5 minutes of quiet time after a heavy conversation.",
    },
    {
        "name": "Logic",
        "strength_title": "Analytical Rigor",
        "overall": "You break complex issues into logical, manageable pieces.",
        "best": "Objective, rational, and excellent at troubleshooting.",
        "watch": "May overlook the emotional impact of a purely logical decision.",
        "action": "Before presenting a solution, consider how it makes others feel.",
    },
    {
        "name": "Focus",
        "strength_title": "Laser Concentration",
        "overall": "You have the ability to dive deep into a single task for extended periods.",
        "best": "Productive, detail-oriented, and resilient against distractions.",
        "watch": "Can develop tunnel vision and miss the broader context.",
        "action": "Set an alarm to step back and look at the big picture every hour.",
    },
]

TRAIT_LOOKUP = {t["name"]: t for t in ALL_TRAITS}


def _compute_stable_seed(text: str) -> int:
    """Generate a deterministic 32-bit integer seed from text bytes."""
    if not text:
        return 42
    digest = hashlib.sha256(text.strip().encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def _calculate_angle(point1: List[float], point2: List[float]) -> float:
    """Compute slope angle in degrees between two 2D points."""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return math.degrees(math.atan2(dy, dx))


def _extract_box_metrics(boxes: List[Any]) -> Dict[str, float]:
    """Safely extract bounding box geometry statistics from OCR output."""
    if not boxes:
        return {"avg_angle": 0.0, "height_consistency": 0.5, "spacing_consistency": 0.5}

    angles: List[float] = []
    heights: List[float] = []
    x_starts: List[float] = []

    for item in boxes:
        if not item or not isinstance(item, (list, tuple)) or len(item) == 0:
            continue
        pts = item[0]
        if not isinstance(pts, (list, tuple)) or len(pts) < 4:
            continue

        try:
            p0, p1, p3 = pts[0], pts[1], pts[3]
            angles.append(_calculate_angle(p0, p1))
            height = abs(p3[1] - p0[1])
            if height > 2:
                heights.append(height)
            x_starts.append(p0[0])
        except (IndexError, TypeError):
            continue

    avg_angle = float(np.mean(angles)) if angles else 0.0

    # Normalized consistency ratios [0.0 - 1.0]
    if len(heights) > 1:
        mean_h = np.mean(heights)
        cv_h = np.std(heights) / (mean_h + 1e-5)
        height_consistency = float(np.clip(1.0 - cv_h, 0.0, 1.0))
    else:
        height_consistency = 0.5

    if len(x_starts) > 2:
        x_starts.sort()
        gaps = np.diff(x_starts)
        valid_gaps = [g for g in gaps if g > 5]
        if valid_gaps:
            cv_g = np.std(valid_gaps) / (np.mean(valid_gaps) + 1e-5)
            spacing_consistency = float(np.clip(1.0 - cv_g, 0.0, 1.0))
        else:
            spacing_consistency = 0.5
    else:
        spacing_consistency = 0.5

    return {
        "avg_angle": avg_angle,
        "height_consistency": height_consistency,
        "spacing_consistency": spacing_consistency,
    }


def generate_profile(
    boxes: Optional[List[Any]] = None,
    text: str = "",
    features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate an explainable HPI assessment profile.

    Parameters
    ----------
    boxes: Bounding boxes from OCR engine
    text: Transcribed handwritten text
    features: Optional dictionary from CV advanced analysis (slant, spacing, shapes, etc.)

    Returns
    -------
    dict with keys: 'standout_strength', 'scores', 'insights'
    """
    boxes = boxes or []
    seed = _compute_stable_seed(text)
    box_metrics = _extract_box_metrics(boxes)

    # 1. Evaluate biometric handwriting attributes
    slant_str = (features.get("slant") if features else "").lower()
    stroke_str = (features.get("stroke_geometry") if features else "").lower()
    spacing_str = (features.get("spacing") if features else "").lower()

    is_forward_slant = "forward" in slant_str or box_metrics["avg_angle"] > 5.0
    is_high_pressure = "heavy" in stroke_str or "high" in stroke_str
    is_wide_spacing = "wide" in spacing_str or box_metrics["spacing_consistency"] < 0.4
    is_methodical = box_metrics["height_consistency"] > 0.65

    # 2. Graphological trait scoring (calibrated between 78 and 98)
    trait_weights: Dict[str, float] = {
        "Execution": 84.0 + (6.0 if is_high_pressure else 0.0) + (4.0 if is_methodical else 0.0),
        "Focus": 82.0 + (7.0 if is_methodical else 2.0) + (3.0 if not is_wide_spacing else -1.0),
        "Logic": 83.0 + (5.0 if is_methodical else 0.0) + (3.0 if not is_forward_slant else 0.0),
        "Communication": 82.0 + (6.0 if is_forward_slant else 0.0) + (4.0 if is_wide_spacing else 0.0),
        "Empathy": 80.0 + (7.0 if is_forward_slant else 0.0) + (3.0 if not is_high_pressure else 0.0),
        "Decision Style": 83.0 + (5.0 if is_high_pressure else 1.0) + (4.0 if not is_forward_slant else 0.0),
        "Adaptability": 81.0 + (6.0 if is_wide_spacing else 0.0) + (4.0 if not is_methodical else 0.0),
        "Creativity": 81.0 + (7.0 if not is_methodical else 0.0) + (4.0 if is_wide_spacing else 0.0),
    }

    # 3. Deterministically rank and select top 5 traits
    sorted_traits = sorted(
        ALL_TRAITS,
        key=lambda t: (
            trait_weights[t["name"]]
            + ((seed >> (ALL_TRAITS.index(t) * 3)) % 5) * 0.5
        ),
        reverse=True,
    )
    selected_traits = sorted_traits[:5]

    scores: List[Dict[str, Any]] = []
    insights: List[Dict[str, Any]] = []

    for i, trait in enumerate(selected_traits):
        name = trait["name"]
        raw_val = trait_weights[name]
        jitter = ((seed >> (i * 4)) % 6) - 2
        final_val = int(np.clip(round(raw_val + jitter), 76, 98))

        scores.append({
            "name": name,
            "value": final_val,
            "strength_title": trait["strength_title"],
        })
        insights.append({
            "title": name,
            "overall_interpretation": trait["overall"],
            "at_your_best": trait["best"],
            "what_to_watch": trait["watch"],
            "action_to_practise": trait["action"],
        })

    standout_trait = max(scores, key=lambda x: x["value"])
    final_scores = [{"name": s["name"], "value": s["value"]} for s in scores]

    return {
        "standout_strength": standout_trait["strength_title"],
        "scores": final_scores,
        "insights": insights,
    }

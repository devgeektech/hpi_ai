"""HPI Assessment Profile Generator.

Centralized engine dynamically mapping physical handwriting measurements (slant,
pressure, spacing, uniformity) into psychological HPI traits and behavioral insights.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from central_ai.core.config import DATA_DIR
from central_ai.core.logging import logger

ALL_TRAITS: List[str] = [
    "Execution",
    "Focus",
    "Logic",
    "Communication",
    "Empathy",
    "Decision Style",
    "Adaptability",
    "Creativity",
]

# Continuous Active Learning Storage Paths
TRAINING_DIR = DATA_DIR / "training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_FILE = TRAINING_DIR / "labeled_samples.jsonl"
MODEL_FILE = DATA_DIR / "models" / "hpi_learned_weights.json"
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

_cached_learned_weights: Optional[Dict[str, Any]] = None


def get_learned_weights() -> Optional[Dict[str, Any]]:
    """Fetch learned ML weights from disk if trained samples exist."""
    global _cached_learned_weights
    if _cached_learned_weights is None and MODEL_FILE.exists():
        try:
            with open(MODEL_FILE, "r", encoding="utf-8") as f:
                _cached_learned_weights = json.load(f)
        except Exception as exc:
            logger.warning(f"Could not read learned weights: {exc}")
    return _cached_learned_weights


def extract_feature_vector(
    slant_deg: float,
    slant_std: float,
    pressure_val: float,
    aspect_ratio: float,
    letter_h: float,
    letter_h_std: float,
    word_gap: float,
    uniformity: float,
) -> List[float]:
    """Normalize physical metrics into an 8-dimensional ML feature vector."""
    return [
        float(slant_deg) / 45.0,
        float(slant_std) / 30.0,
        float(pressure_val) / 0.10,
        float(aspect_ratio) / 1.5,
        float(letter_h) / 60.0,
        float(letter_h_std) / 25.0,
        float(word_gap) / 50.0,
        float(uniformity),
    ]


def record_and_train_sample(
    features: Dict[str, Any],
    verified_scores: Dict[str, float],
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Side-by-side active learning: records verified sample and retrains model instantly."""
    global _cached_learned_weights
    metrics = (features.get("metrics") or {}) if features else {}

    vec = extract_feature_vector(
        slant_deg=float(metrics.get("slant_angle", 0.0)),
        slant_std=float(metrics.get("slant_std", 15.0)),
        pressure_val=float(metrics.get("pressure_density", 0.04)),
        aspect_ratio=float(metrics.get("aspect_ratio_mean", 1.0)),
        letter_h=float(metrics.get("letter_height_mean", 30.0)),
        letter_h_std=float(metrics.get("letter_height_std", 10.0)),
        word_gap=float(metrics.get("word_gap_mean", 25.0)),
        uniformity=float(metrics.get("uniformity_score", 0.5)),
    )

    clean_scores = {k: float(v) for k, v in verified_scores.items() if k in ALL_TRAITS}
    if not clean_scores:
        return {"status": "skipped", "reason": "No recognized HPI trait scores provided"}

    record = {
        "sample_id": sample_id or f"SMP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "vector": vec,
        "scores": clean_scores,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    # Append to labeled training data
    with open(SAMPLES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Read all labeled samples and retrain Ridge regression
    all_vectors = []
    all_targets = []
    with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                all_vectors.append(item["vector"])
                all_targets.append([item["scores"].get(t, 80.0) for t in ALL_TRAITS])

    X = np.array(all_vectors, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)

    # Regularized Ridge regression: W = (X_b^T X_b + alpha * I)^(-1) X_b^T y
    X_bias = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
    alpha = 1.0  # Ridge penalty ensures mathematical stability even for small N
    identity = np.eye(X_bias.shape[1], dtype=np.float32)
    identity[0, 0] = 0.0  # Do not regularize bias offset
    weights_matrix = np.linalg.solve(X_bias.T @ X_bias + alpha * identity, X_bias.T @ y)

    model_data = {
        "model_version": "hpi-active-learning-v1",
        "total_samples_trained": len(all_vectors),
        "traits": ALL_TRAITS,
        "bias": weights_matrix[0].tolist(),
        "weights": weights_matrix[1:].tolist(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)

    _cached_learned_weights = model_data
    logger.info(
        f"[Active Learning] Retrained HPI behavioral model on {len(all_vectors)} sample(s). "
        f"Weights updated at {MODEL_FILE}"
    )

    return {
        "status": "trained",
        "total_samples_trained": len(all_vectors),
        "updated_at": model_data["updated_at"],
    }


def get_model_status() -> Dict[str, Any]:
    """Inspect current active learning model state."""
    learned = get_learned_weights()
    if learned:
        return {
            "mode": "trained_supervised_ml",
            "total_samples_trained": learned.get("total_samples_trained", 0),
            "last_updated": learned.get("updated_at"),
            "model_version": learned.get("model_version", "hpi-active-learning-v1"),
        }
    return {
        "mode": "calibrated_graphological_rules",
        "total_samples_trained": 0,
        "last_updated": None,
        "model_version": "hpi-v1-graphological-baseline",
    }


def _calculate_angle(point1: List[float], point2: List[float]) -> float:
    """Compute slope angle in degrees between two 2D points."""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return math.degrees(math.atan2(dy, dx))


def _extract_box_metrics(boxes: List[Any]) -> Dict[str, float]:
    """Extract bounding box geometry statistics from OCR output."""
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


def _build_dynamic_insight(
    trait_name: str,
    score: int,
    slant_deg: float,
    pressure_val: float,
    word_gap: float,
    uniformity: float,
) -> Dict[str, str]:
    """Dynamically generate behavioral interpretation based on measured handwriting attributes."""
    if trait_name == "Execution":
        if pressure_val > 0.08:
            return {
                "title": "Structured Execution",
                "overall": "You drive projects to completion with intense conviction, high stamina, and decisive follow-through.",
                "best": "Highly reliable, determined under pressure, and organized.",
                "watch": "Can become impatient when collaborators move at a slower pace.",
                "action": "Build deliberate 15-minute buffer blocks into your daily schedule.",
            }
        else:
            return {
                "title": "Methodical Execution",
                "overall": "You approach tasks steadily with thoughtful pacing and consistent attention to detail.",
                "best": "Punctual, thorough, and careful with commitments.",
                "watch": "May hesitate to initiate action without complete guidelines.",
                "action": "Set a rapid 24-hour target for your next project milestone.",
            }

    elif trait_name == "Communication":
        if slant_deg > 8.0:
            return {
                "title": "Expressive Connection",
                "overall": "Your communication is naturally warm, relational, and engaging, encouraging open team dialogue.",
                "best": "Empathetic, spontaneous, and easily establishes rapport.",
                "watch": "Can occasionally prioritize harmony over difficult feedback.",
                "action": "Anchor conversations with one direct, clear question before sharing perspectives.",
            }
        else:
            return {
                "title": "Objective Clarity",
                "overall": "You articulate thoughts with calm composure, clarity, and rational structure.",
                "best": "Clear, grounded, and inspires trust in high-stakes discussions.",
                "watch": "Can appear reserved or understated in casual team settings.",
                "action": "Open important updates with personal context to build closer connection.",
            }

    elif trait_name == "Logic":
        return {
            "title": "Analytical Rigor",
            "overall": "You dissect complex problems into clear, manageable principles before committing resources.",
            "best": "Objective, rational, and exceptionally strong at structured troubleshooting.",
            "watch": "May overlook emotional factors that influence team buy-in.",
            "action": "Evaluate how key decisions will impact team sentiment before implementation.",
        }

    elif trait_name == "Creativity":
        return {
            "title": "Innovative Thinking",
            "overall": "You approach challenges with original angles, connecting ideas across disparate domains.",
            "best": "Resourceful, imaginative, and comfortable venturing beyond standard procedures.",
            "watch": "Can generate more concepts than can be practically executed simultaneously.",
            "action": "Select your single highest-potential idea each week and define three immediate steps.",
        }

    elif trait_name == "Focus":
        return {
            "title": "Laser Concentration",
            "overall": "You maintain deep concentration on critical priorities, filtering out non-essential distractions.",
            "best": "Productive, detail-oriented, and thorough in deep work sessions.",
            "watch": "Can develop tunnel vision and temporarily overlook broader strategic shifts.",
            "action": "Schedule an hourly reminder to step back and evaluate the big picture.",
        }

    elif trait_name == "Empathy":
        return {
            "title": "Deep Intuition",
            "overall": "You readily perceive interpersonal dynamics and the unspoken emotional tone of group interactions.",
            "best": "Supportive, attentive listener, and fosters inclusive team environments.",
            "watch": "Can absorb others' anxiety or stress during tense periods.",
            "action": "Dedicate five minutes of quiet time to recharge following emotionally demanding meetings.",
        }

    elif trait_name == "Decision Style":
        if word_gap > 28.0:
            return {
                "title": "Strategic Deliberation",
                "overall": "You evaluate options with patience and independent foresight, preserving strong analytical boundaries.",
                "best": "Visionary, prudent, and avoids rash decisions.",
                "watch": "Can delay choices while seeking additional data.",
                "action": "Trust your first analytical assessment for decisions with low downside risk.",
            }
        else:
            return {
                "title": "Decisive Momentum",
                "overall": "You make rapid, decisive choices to maintain forward project momentum.",
                "best": "Action-oriented, confident, and responsive.",
                "watch": "May skip checking secondary edge cases under tight deadlines.",
                "action": "Take a 60-second sanity check before finalizing major commitments.",
            }

    elif trait_name == "Adaptability":
        return {
            "title": "Adaptive Agility",
            "overall": "You pivot smoothly when new data emerges, responding constructively to changing conditions.",
            "best": "Resilient, flexible, and comfortable in ambiguity.",
            "watch": "Shifting priorities too frequently can dilute long-term focus.",
            "action": "Lock in one non-negotiable core goal each week to anchor flexibility.",
        }

    return {
        "title": trait_name,
        "overall": f"Demonstrates balanced dynamic capacity in {trait_name.lower()}.",
        "best": "Consistent and thoughtful.",
        "watch": "Monitor balance across fluctuating demands.",
        "action": "Maintain reflective check-ins on weekly performance.",
    }


def generate_profile(
    boxes: Optional[List[Any]] = None,
    text: str = "",
    features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate an explainable HPI assessment profile purely from measured physical handwriting attributes.

    Parameters
    ----------
    boxes: Bounding boxes from OCR engine
    text: Transcribed handwritten text
    features: Dictionary from CV advanced analysis containing physical metrics (slant, pressure, spacing, etc.)

    Returns
    -------
    dict with keys: 'standout_strength', 'scores', 'insights'
    """
    boxes = boxes or []
    box_metrics = _extract_box_metrics(boxes)

    # 1. Continuous physical measurements extracted from OpenCV analysis
    metrics = (features.get("metrics") or {}) if features else {}
    slant_deg = float(metrics.get("slant_angle", box_metrics.get("avg_angle", 0.0)))
    slant_std = float(metrics.get("slant_std", 15.0))
    pressure_val = float(metrics.get("pressure_density", 0.04))
    word_gap = float(metrics.get("word_gap_mean", 25.0))
    uniformity = float(metrics.get("uniformity_score", box_metrics.get("height_consistency", 0.5)))
    aspect_ratio = float(metrics.get("aspect_ratio_mean", 1.0))
    letter_h = float(metrics.get("letter_height_mean", 30.0))
    letter_h_std = float(metrics.get("letter_height_std", 10.0))

    # 2. Check if active learning trained model is available
    learned = get_learned_weights()
    if learned and "bias" in learned and "weights" in learned:
        vec = np.array(
            extract_feature_vector(
                slant_deg=slant_deg,
                slant_std=slant_std,
                pressure_val=pressure_val,
                aspect_ratio=aspect_ratio,
                letter_h=letter_h,
                letter_h_std=letter_h_std,
                word_gap=word_gap,
                uniformity=uniformity,
            ),
            dtype=np.float32,
        )
        bias = np.array(learned["bias"], dtype=np.float32)
        weights = np.array(learned["weights"], dtype=np.float32)
        predicted_vals = bias + (vec @ weights)
        trait_names_list = learned.get("traits", ALL_TRAITS)
        trait_weights: Dict[str, float] = {
            t: float(predicted_vals[idx]) for idx, t in enumerate(trait_names_list)
        }
    else:
        # Grounded in physical measurements via calibrated graphological baseline
        trait_weights = {
            # Execution: Driven by firm ink pressure, disciplined line consistency, and baseline steadiness
            "Execution": 78.0 + min(12.0, (pressure_val / 0.08) * 8.0) + (uniformity * 8.0),

            # Focus: Driven by precise letter height, high uniformity, and disciplined word spacing
            "Focus": 79.0 + (uniformity * 9.0) + max(0.0, min(8.0, (40.0 - letter_h) * 0.3)),

            # Logic: Driven by upright/neutral slant (-6° to +14°), structured word gaps, and methodical consistency
            "Logic": 80.0 + (6.0 if -6.0 <= slant_deg <= 14.0 else 1.0) + min(7.0, (word_gap / 30.0) * 5.0) + (uniformity * 4.0),

            # Communication: Driven by forward slant (expressive social orientation) and broad open letter curves
            "Communication": 78.0 + min(11.0, max(-3.0, (slant_deg / 20.0) * 8.0)) + min(7.0, (aspect_ratio / 1.2) * 5.0),

            # Empathy: Driven by rightward slant (emotional sensitivity) and receptive, non-rigid stroke pressure
            "Empathy": 77.0 + min(13.0, max(-2.0, (slant_deg / 18.0) * 9.0)) + (5.0 if pressure_val < 0.07 else 1.0),

            # Decision Style: Driven by deliberate stroke pressure, clear boundary spacing, and upright stance
            "Decision Style": 80.0 + min(10.0, (pressure_val / 0.08) * 6.0) + min(6.0, (word_gap / 25.0) * 4.0) + (uniformity * 4.0),

            # Adaptability: Driven by open letter curves, flexible spacing, and dynamic baseline flow
            "Adaptability": 79.0 + min(9.0, (aspect_ratio / 1.2) * 6.0) + (6.0 if uniformity < 0.75 else 2.0),

            # Creativity: Driven by loop expansiveness, slant dynamism, and non-rigid stroke variation
            "Creativity": 78.0 + min(10.0, (aspect_ratio / 1.2) * 6.0) + (6.0 if uniformity < 0.70 else 1.0) + min(4.0, (letter_h / 40.0) * 4.0),
        }

    # 3. Dynamically rank all 8 traits based on exact measured scores
    ranked_trait_names = sorted(
        trait_weights.keys(),
        key=lambda name: trait_weights[name],
        reverse=True,
    )

    scores: List[Dict[str, Any]] = []
    insights: List[Dict[str, Any]] = []

    # Select top 5 standout traits
    for trait_name in ranked_trait_names[:5]:
        final_val = int(np.clip(round(trait_weights[trait_name]), 70, 99))
        insight_info = _build_dynamic_insight(
            trait_name=trait_name,
            score=final_val,
            slant_deg=slant_deg,
            pressure_val=pressure_val,
            word_gap=word_gap,
            uniformity=uniformity,
        )

        scores.append({
            "name": trait_name,
            "value": final_val,
            "strength_title": insight_info["title"],
        })
        insights.append({
            "title": trait_name,
            "overall_interpretation": insight_info["overall"],
            "at_your_best": insight_info["best"],
            "what_to_watch": insight_info["watch"],
            "action_to_practise": insight_info["action"],
        })

    standout_trait = max(scores, key=lambda x: x["value"])
    final_scores = [{"name": s["name"], "value": s["value"]} for s in scores]

    return {
        "standout_strength": standout_trait["strength_title"],
        "scores": final_scores,
        "insights": insights,
    }

"""Continuous Active Learning Engine for Handwriting Feature Model.

Manages feature vector extraction, verified sample recording, and continuous
regularized Ridge regression updates for handwriting prediction models.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from central_ai.core.config import DATA_DIR
from central_ai.core.logging import logger

SUPPORTED_METRIC_TARGETS: List[str] = [
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
MODEL_FILE = DATA_DIR / "models" / "handwriting_learned_weights.json"
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
    """Normalize physical handwriting metrics into an 8-dimensional ML feature vector."""
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

    clean_scores = {k: float(v) for k, v in verified_scores.items() if k in SUPPORTED_METRIC_TARGETS}
    if not clean_scores:
        return {"status": "skipped", "reason": "No recognized score targets provided"}

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
                all_targets.append([item["scores"].get(t, 80.0) for t in SUPPORTED_METRIC_TARGETS])

    X = np.array(all_vectors, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)

    # Regularized Ridge regression: W = (X_b^T X_b + alpha * I)^(-1) X_b^T y
    X_bias = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
    alpha = 1.0  # Ridge penalty ensures mathematical stability even for small sample sizes
    identity = np.eye(X_bias.shape[1], dtype=np.float32)
    identity[0, 0] = 0.0  # Do not regularize bias offset
    weights_matrix = np.linalg.solve(X_bias.T @ X_bias + alpha * identity, X_bias.T @ y)

    model_data = {
        "model_version": "hw-active-learning-v1",
        "total_samples_trained": len(all_vectors),
        "targets": SUPPORTED_METRIC_TARGETS,
        "bias": weights_matrix[0].tolist(),
        "weights": weights_matrix[1:].tolist(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)

    _cached_learned_weights = model_data
    logger.info(
        f"[Active Learning] Retrained handwriting model on {len(all_vectors)} sample(s). "
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
            "model_version": learned.get("model_version", "hw-active-learning-v1"),
        }
    return {
        "mode": "baseline_active_learning",
        "total_samples_trained": 0,
        "last_updated": None,
        "model_version": "hw-v1-baseline",
    }

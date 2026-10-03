"""Central AI configuration settings and runtime constants."""

from __future__ import annotations

import os
from pathlib import Path

# Project directory paths
PACKAGE_DIR = Path(__file__).resolve().parent.parent  # src/central_ai
ROOT_DIR = PACKAGE_DIR.parent.parent                  # OCR-Project root

DATA_DIR = Path(os.getenv("CENTRAL_DATA_DIR", ROOT_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Security & Tenancy
API_KEY: str = os.getenv("CENTRAL_API_KEY", "dev-key")
DEFAULT_PROJECT_ID: str = "hpi"

# Model Configuration
DEFAULT_OCR_ENGINE: str = os.getenv("DEFAULT_OCR_ENGINE", "paddle")
TROCR_MODEL_ID: str = os.getenv("TROCR_MODEL_ID", "microsoft/trocr-base-handwritten")

# Domain Feature Versions
VERSIONS: dict[str, str] = {
    "model_version": "handwriting-v1",
    "methodology_version": "hpi-v1",
    "rule_version": "rules-v1",
    "feature_schema_version": "features-v1",
    "report_version": "report-v1",
    "nlp_version": "nlp-stub-v1",
}

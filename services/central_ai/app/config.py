import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # OCR-Project
DATA_DIR = Path(os.getenv("CENTRAL_DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
(UPLOADS_DIR := DATA_DIR / "uploads").mkdir(parents=True, exist_ok=True)

# PostgreSQL (override with CENTRAL_DATABASE_URL)
DATABASE_URL = os.getenv(
    "CENTRAL_DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/central_ai",
)
API_KEY = os.getenv("CENTRAL_API_KEY", "dev-key")
CONFIDENCE_AUTO_APPROVE = float(os.getenv("CONFIDENCE_AUTO_APPROVE", "85"))

# Central AI — Handwriting Analysis API

Stateless **Central AI** service: upload a handwriting image, get OCR + HPI profile JSON.
**No database. No product UI.** Callers persist results themselves.

Designed as a **centralized microservice** for multiple products via `X-Project-Id`.

## Architecture

```text
Consumer App / Web Client ──POST image──► Central AI (:8000) ──► OCR + HPI Profile JSON
                                          (Stateless, fast, deterministic)
```

## Directory Structure (Antigravity Standard)

```text
OCR-Project/
├── src/
│   └── central_ai/
│       ├── api/              # FastAPI route controllers & tenancy dependencies
│       ├── core/             # Configuration & structured logging
│       ├── engines/          # Preprocessing, PaddleOCR/TrOCR, and HPI assessment
│       ├── schemas/          # Strongly-typed Pydantic V2 data contracts
│       ├── client/           # CentralAIClient Python SDK for consumer apps
│       └── main.py           # Application entry point
├── data/                     # Runtime uploads and logs
│   ├── logs/
│   └── uploads/
├── scripts/                  # Operations and startup scripts
│   └── run_central.ps1
├── pyproject.toml            # Modern Python package specification
└── requirements.txt          # Pinned dependencies
```

## Setup

```powershell
cd "c:\Users\GT49220\OneDrive\Desktop\Sushil-Projects\OCR-Project"
.\venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements.txt
```

Optional environment variables:

```powershell
$env:CENTRAL_API_KEY = "dev-key"
$env:TROCR_MODEL_ID = "microsoft/trocr-base-handwritten"   # default
```

## Run

```powershell
.\scripts\run_central.ps1
```

Or directly via uvicorn:

```powershell
$env:PYTHONPATH = "src"
uvicorn central_ai.main:app --host 127.0.0.1 --port 8000
```

- API root: http://127.0.0.1:8000/
- Swagger docs: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/api/v1/health

## API Usage

### 1. Analyze Handwriting & Generate HPI Profile

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/handwriting/analyze" `
  -H "X-Api-Key: dev-key" `
  -H "X-Project-Id: hpi" `
  -F "file=@note.jpg"
```

Response schema includes:
- `result_id` (str)
- `status` ("completed" | "low_confidence")
- `confidence` (float)
- `features` (`slant`, `spacing`, `shapes`, `stroke_geometry`, `consistency`, `detected_text`)
- `profile` (`standout_strength`, `scores`, `insights`)

### 2. Image Quality Validation Only

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/images/validate" `
  -H "X-Api-Key: dev-key" `
  -H "X-Project-Id: hpi" `
  -F "file=@note.jpg"
```

Response includes `scan_quality`, `handwriting_detected`, and `retake`.

from pathlib import Path
from datetime import datetime
from typing import Optional
import json
import sys
import uuid
import os

import asyncio

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages" / "shared"))
from shared.client import CentralAIClient

DATABASE_URL = os.getenv(
    "HPI_DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/hpi_app",
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Assessment(Base):
    __tablename__ = "assessments"
    id = Column(Integer, primary_key=True)
    central_job_id = Column(String, index=True)
    result_id = Column(String, nullable=True, index=True)
    status = Column(String, default="queued")
    profile_json = Column(Text, nullable=True)
    confidence = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class PrimeSession(Base):
    __tablename__ = "prime_sessions"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, index=True)
    result_id = Column(String, index=True)
    interpretation_json = Column(Text)
    report_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AlignJourney(Base):
    __tablename__ = "align_journeys"
    id = Column(Integer, primary_key=True)
    journey_id = Column(String, unique=True, index=True)
    result_id = Column(String, index=True)
    focus = Column(String, nullable=True)
    plan_json = Column(Text)
    progress_json = Column(Text)
    report_json = Column(Text, nullable=True)
    weeks_complete = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)
app = FastAPI(title="HPI Individual Platform", version="0.1.0")
client = CentralAIClient(project_id="hpi")

PUBLIC = Path(__file__).resolve().parents[1] / "public"
if PUBLIC.is_dir():
    app.mount("/public", StaticFiles(directory=str(PUBLIC)), name="public")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class PrimeCreate(BaseModel):
    result_id: str


class AlignCreate(BaseModel):
    result_id: str
    focus: Optional[str] = None


class AlignProgress(BaseModel):
    weeks_complete: int


def _require_approved_result(result_id: str) -> dict:
    # Find via job listing is not available; apps poll job then store.
    # For PRIME/ALIGN we accept result by re-fetching through a stored assessment.
    # Testing git commits -sandy add again
    return {"result_id": result_id}


@app.get("/")
def root():
    return {
        "service": "hpi",
        "message": "HPI Individual — Assessment, PRIME™, ALIGN™",
        "docs": "/docs",
        "ui": "/public/index.html",
        "paths": {"prime": "/prime", "align": "/align"},
    }


@app.get("/health")
def health():
    try:
        return {"status": "ok", "central_ai": client.health()}
    except Exception as e:
        return {"status": "ok", "central_ai": {"error": str(e)}}


@app.get("/ui")
def ui_redirect():
    return RedirectResponse(url="/public/index.html")


# ----- Legacy UI API (public/app.js expects these paths on the HPI host) -----


@app.post("/notes/analyze/")
async def notes_analyze(file: UploadFile = File(...)):
    data = await file.read()
    try:
        return client.validate_image(data, file.filename or "note.jpg")
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Central AI unavailable on port 8000. Start it first. ({e})",
        ) from e


@app.post("/notes/advanced_analysis/")
async def notes_advanced_analysis(file: UploadFile = File(...)):
    data = await file.read()
    try:
        return client.handwriting_features(data, file.filename or "note.jpg")
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Central AI unavailable on port 8000. Start it first. ({e})",
        ) from e


@app.post("/notes/upload_stream/")
async def notes_upload_stream(
    file: UploadFile = File(...),
):
    # Read BEFORE StreamingResponse — UploadFile can close once the response starts
    image_bytes = await file.read()
    filename = file.filename or "note.jpg"

    async def event_generator():
        db = None
        try:
            # Yield immediately so the client gets the first SSE frame before any I/O
            yield f"data: {json.dumps({'status': 'Preparing document...'})}\n\n"
            await asyncio.sleep(0.5)

            # Open DB only after the stream has started (Depends(db) is unsafe with SSE)
            db = SessionLocal()
            yield f"data: {json.dumps({'status': 'Identifying handwriting characteristics...'})}\n\n"

            try:
                job = await asyncio.to_thread(
                    client.analyze_handwriting,
                    image_bytes,
                    filename,
                    "image/jpeg",
                    "hpi-web",
                )
            except Exception as e:
                yield f"data: {json.dumps({'status': 'Failed', 'error': str(e)})}\n\n"
                return

            job_id = job["job_id"]
            row = Assessment(central_job_id=job_id, status="queued")
            db.add(row)
            db.commit()
            db.refresh(row)

            job_status = {"status": "queued"}
            for _ in range(900):
                await asyncio.sleep(1)
                try:
                    job_status = await asyncio.to_thread(client.get_job, job_id)
                except Exception as e:
                    yield f"data: {json.dumps({'status': 'Failed', 'error': str(e)})}\n\n"
                    return
                status = job_status.get("status")
                if status in ("completed", "needs_review", "failed"):
                    break

            yield f"data: {json.dumps({'status': 'Building your profile...'})}\n\n"
            await asyncio.sleep(0.5)
            yield f"data: {json.dumps({'status': 'Preparing personalised insights...'})}\n\n"
            await asyncio.sleep(0.5)

            if job_status.get("status") == "failed":
                yield f"data: {json.dumps({'status': 'Failed', 'message': job_status.get('message')})}\n\n"
                return

            try:
                result = await asyncio.to_thread(client.get_job_result, job_id)
            except Exception as e:
                yield f"data: {json.dumps({'status': 'Failed', 'error': str(e)})}\n\n"
                return

            features = result.get("features") or {}
            profile = result.get("profile") or {}
            row.result_id = result.get("result_id")
            row.status = result.get("status", job_status.get("status"))
            row.profile_json = json.dumps(profile)
            row.confidence = float(result.get("confidence") or 0)
            db.commit()

            final_response = {
                "scan_quality": features.get("scan_quality", "unknown"),
                "handwriting_detected": features.get("handwriting_detected", 0),
                "retake": features.get("retake", False),
                "detected_text": features.get("detected_text", ""),
                "profile": profile,
            }
            yield f"data: {json.dumps({'status': 'Complete', 'result': final_response})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'status': 'Failed', 'error': str(e)})}\n\n"
        finally:
            if db is not None:
                db.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/assessments/analyze")
async def start_assessment(file: UploadFile = File(...), db: Session = Depends(get_db)):
    data = await file.read()
    quality = client.validate_image(data, file.filename or "note.jpg")
    if quality.get("retake"):
        return {"retake": True, "quality": quality}
    job = client.analyze_handwriting(data, file.filename or "note.jpg", device_id="hpi-web")
    row = Assessment(central_job_id=job["job_id"], status="queued")
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"assessment_id": row.id, "job_id": job["job_id"], "quality": quality}


@app.get("/assessments/{assessment_id}")
def get_assessment(assessment_id: int, db: Session = Depends(get_db)):
    row = db.query(Assessment).filter(Assessment.id == assessment_id).first()
    if not row:
        raise HTTPException(404, "Assessment not found")
    job = client.get_job(row.central_job_id)
    if job.get("result_id"):
        result = client.get_job_result(row.central_job_id)
        row.result_id = result["result_id"]
        row.status = result.get("status", job.get("status"))
        row.profile_json = json.dumps(result.get("profile") or {})
        row.confidence = float(result.get("confidence") or 0)
        db.commit()
        return {"assessment": {"id": row.id, "status": row.status, "result_id": row.result_id}, "job": job, "result": result}
    return {"assessment": {"id": row.id, "status": row.status}, "job": job}


# ----- PRIME™ -----

@app.post("/prime/sessions")
def create_prime(body: PrimeCreate, db: Session = Depends(get_db)):
    assessment = (
        db.query(Assessment)
        .filter(Assessment.result_id == body.result_id)
        .order_by(Assessment.id.desc())
        .first()
    )
    if not assessment or assessment.status not in ("approved", "completed"):
        # Allow needs_review blocked
        if assessment and assessment.status == "needs_review":
            raise HTTPException(400, "Result needs human review before PRIME")
        if not assessment:
            raise HTTPException(404, "Unknown result_id — complete an assessment first")

    profile = json.loads(assessment.profile_json or "{}")
    interpretation = {
        "path": "PRIME",
        "standout_strength": profile.get("standout_strength"),
        "scores": profile.get("scores", []),
        "insights": profile.get("insights", []),
        "summary": f"PRIME™ interpretation centered on {profile.get('standout_strength', 'your profile')}.",
        "evidence_note": "Derived only from approved Central AI HPI profile.",
    }
    session_id = f"PRIME-{uuid.uuid4().hex[:8].upper()}"
    row = PrimeSession(
        session_id=session_id,
        result_id=body.result_id,
        interpretation_json=json.dumps(interpretation),
    )
    db.add(row)
    db.commit()
    return {"session_id": session_id, "interpretation": interpretation}


@app.get("/prime/sessions/{session_id}")
def get_prime(session_id: str, db: Session = Depends(get_db)):
    row = db.query(PrimeSession).filter(PrimeSession.session_id == session_id).first()
    if not row:
        raise HTTPException(404, "PRIME session not found")
    return {
        "session_id": row.session_id,
        "result_id": row.result_id,
        "interpretation": json.loads(row.interpretation_json or "{}"),
        "report": json.loads(row.report_json) if row.report_json else None,
    }


@app.post("/prime/sessions/{session_id}/reports/generate")
def prime_report(session_id: str, db: Session = Depends(get_db)):
    row = db.query(PrimeSession).filter(PrimeSession.session_id == session_id).first()
    if not row:
        raise HTTPException(404, "PRIME session not found")
    interpretation = json.loads(row.interpretation_json or "{}")
    report = {
        "report_type": "prime",
        "report_version": "report-v1",
        "title": "PRIME™ Interpretation Report",
        "standout_strength": interpretation.get("standout_strength"),
        "insights": interpretation.get("insights", []),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    row.report_json = json.dumps(report)
    db.commit()
    return report


# ----- ALIGN™ -----

DEFAULT_WEEKS = [
    {"week_number": 1, "title": "Self-awareness", "description": "Understand your natural patterns.", "status": "active"},
    {"week_number": 2, "title": "Communication", "description": "How you express and connect.", "status": "locked"},
    {"week_number": 3, "title": "Adaptability", "description": "Respond to change and uncertainty.", "status": "locked"},
    {"week_number": 4, "title": "Applied Practice", "description": "Put insights into everyday action.", "status": "locked"},
]


@app.post("/align/journeys")
def create_align(body: AlignCreate, db: Session = Depends(get_db)):
    assessment = (
        db.query(Assessment)
        .filter(Assessment.result_id == body.result_id)
        .order_by(Assessment.id.desc())
        .first()
    )
    if not assessment:
        raise HTTPException(404, "Unknown result_id — complete an assessment first")
    if assessment.status == "needs_review":
        raise HTTPException(400, "Result needs human review before ALIGN")
    if assessment.status not in ("approved", "completed"):
        raise HTTPException(400, f"Result status '{assessment.status}' not ready for ALIGN")

    profile = json.loads(assessment.profile_json or "{}")
    focus = body.focus or profile.get("standout_strength") or "Adaptive Communication"
    plan = {
        "path": "ALIGN",
        "current_focus": focus,
        "total_weeks": 4,
        "weeks": DEFAULT_WEEKS,
    }
    journey_id = f"ALIGN-{uuid.uuid4().hex[:8].upper()}"
    row = AlignJourney(
        journey_id=journey_id,
        result_id=body.result_id,
        focus=focus,
        plan_json=json.dumps(plan),
        progress_json=json.dumps({"weeks_complete": 0}),
        weeks_complete=0,
    )
    db.add(row)
    db.commit()
    return {"journey_id": journey_id, "plan": plan}


@app.get("/align/journeys/{journey_id}")
def get_align(journey_id: str, db: Session = Depends(get_db)):
    row = db.query(AlignJourney).filter(AlignJourney.journey_id == journey_id).first()
    if not row:
        raise HTTPException(404, "ALIGN journey not found")
    plan = json.loads(row.plan_json or "{}")
    weeks = plan.get("weeks", [])
    for w in weeks:
        n = w["week_number"]
        if row.weeks_complete >= n:
            w["status"] = "completed"
        elif row.weeks_complete == n - 1:
            w["status"] = "active"
        else:
            w["status"] = "locked"
    return {
        "journey_id": row.journey_id,
        "result_id": row.result_id,
        "focus": row.focus,
        "weeks_complete": row.weeks_complete,
        "weeks": weeks,
        "report": json.loads(row.report_json) if row.report_json else None,
    }


@app.post("/align/journeys/{journey_id}/progress")
def align_progress(journey_id: str, body: AlignProgress, db: Session = Depends(get_db)):
    row = db.query(AlignJourney).filter(AlignJourney.journey_id == journey_id).first()
    if not row:
        raise HTTPException(404, "ALIGN journey not found")
    row.weeks_complete = max(0, min(4, body.weeks_complete))
    row.progress_json = json.dumps({"weeks_complete": row.weeks_complete})
    db.commit()
    return get_align(journey_id, db)


@app.post("/align/journeys/{journey_id}/reports/generate")
def align_report(journey_id: str, db: Session = Depends(get_db)):
    row = db.query(AlignJourney).filter(AlignJourney.journey_id == journey_id).first()
    if not row:
        raise HTTPException(404, "ALIGN journey not found")
    report = {
        "report_type": "align",
        "report_version": "report-v1",
        "title": "ALIGN™ Development Report",
        "focus": row.focus,
        "weeks_complete": row.weeks_complete,
        "total_weeks": 4,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    row.report_json = json.dumps(report)
    db.commit()
    return report


@app.get("/paths", response_class=HTMLResponse)
def paths_ui():
    return """
    <html><body style="font-family:sans-serif;max-width:720px;margin:2rem auto">
      <h1>HPI™ Individual</h1>
      <p>After an approved assessment, choose a path:</p>
      <ul>
        <li><b>PRIME™</b> — Interpretation of your HPI profile</li>
        <li><b>ALIGN™</b> — 4-week development journey</li>
      </ul>
      <p>Assessment UI: <a href="/public/index.html">/public/index.html</a></p>
      <p>API: <a href="/docs">/docs</a></p>
    </body></html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.hpi.app.main:app", host="0.0.0.0", port=8003, reload=True)

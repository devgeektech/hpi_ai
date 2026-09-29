import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages" / "shared"))

from shared.client import CentralAIClient

DATABASE_URL = os.getenv(
    "HMB_DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/hmb",
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class ExerciseSubmission(Base):
    __tablename__ = "exercise_submissions"
    id = Column(Integer, primary_key=True)
    exercise_id = Column(String, default="ex-baseline-01")
    central_job_id = Column(String, nullable=True)
    skill_score = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

app = FastAPI(title="HMB — Handwriting MetaSkills Builder", version="0.1.0")
client = CentralAIClient(project_id="hmb")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {
        "service": "hmb",
        "message": "Handwriting MetaSkills Builder",
        "docs": "/docs",
        "ui": "/ui",
    }


@app.get("/health")
def health():
    central = {}
    try:
        central = client.health()
    except Exception as e:
        central = {"error": str(e)}
    return {"status": "ok", "central_ai": central}


@app.post("/exercises/submit")
async def submit_exercise(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    data = await file.read()
    quality = client.validate_image(data, file.filename or "note.jpg")
    if quality.get("retake"):
        return {"retake": True, "quality": quality}

    job = client.analyze_handwriting(data, file.filename or "note.jpg", device_id="hmb-web")
    row = ExerciseSubmission(
        exercise_id="ex-baseline-01",
        central_job_id=job["job_id"],
        skill_score=0,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {
        "submission_id": row.id,
        "job_id": job["job_id"],
        "quality": quality,
        "message": "Queued on Central AI — poll /exercises/status/{job_id}",
    }


@app.get("/exercises/status/{job_id}")
def exercise_status(job_id: str, db: Session = Depends(get_db)):
    job = client.get_job(job_id)
    out = {"job": job}
    if job.get("status") in ("completed", "needs_review") and job.get("result_id"):
        result = client.get_job_result(job_id)
        # Simple skill score from confidence
        score = float(result.get("confidence") or 0)
        row = db.query(ExerciseSubmission).filter(ExerciseSubmission.central_job_id == job_id).first()
        if row:
            row.skill_score = score
            db.commit()
        out["skill_score"] = score
        out["result"] = result
        out["feedback"] = {
            "strengths": ["Detected stable handwriting sample"],
            "improvements": ["Continue baseline and spacing drills"],
        }
    return out


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <html><head><title>HMB</title></head>
    <body style="font-family:sans-serif;max-width:640px;margin:2rem auto">
      <h1>HMB™</h1>
      <p>Upload an exercise photo. Analysis runs on Central AI (:8000).</p>
      <input type="file" id="f" accept="image/*"/>
      <button onclick="go()">Submit</button>
      <pre id="out"></pre>
      <script>
        async function go(){
          const f=document.getElementById('f').files[0];
          const fd=new FormData(); fd.append('file', f);
          const r=await fetch('/exercises/submit',{method:'POST',body:fd});
          document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2);
        }
      </script>
    </body></html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.hmb.app.main:app", host="0.0.0.0", port=8001, reload=True)

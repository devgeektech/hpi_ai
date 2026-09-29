from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import Column, Integer, String, DateTime, create_engine, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from datetime import datetime
import json
import sys
import os

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages" / "shared"))
from shared.client import CentralAIClient

DATABASE_URL = os.getenv(
    "BOARDROOM_DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/boardroom",
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Participant(Base):
    __tablename__ = "participants"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    central_job_id = Column(String, nullable=True)
    result_id = Column(String, nullable=True)
    profile_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)
app = FastAPI(title="Boardroom Breakthroughs", version="0.1.0")
client = CentralAIClient(project_id="boardroom")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {"service": "boardroom", "docs": "/docs", "ui": "/ui"}


@app.get("/health")
def health():
    try:
        return {"status": "ok", "central_ai": client.health()}
    except Exception as e:
        return {"status": "ok", "central_ai": {"error": str(e)}}


@app.post("/participants")
def create_participant(name: str, db: Session = Depends(get_db)):
    p = Participant(name=name)
    db.add(p)
    db.commit()
    db.refresh(p)
    return {"id": p.id, "name": p.name}


@app.post("/participants/{participant_id}/analyze")
async def analyze_participant(
    participant_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    p = db.query(Participant).filter(Participant.id == participant_id).first()
    if not p:
        return {"error": "Participant not found"}
    data = await file.read()
    job = client.analyze_handwriting(data, file.filename or "note.jpg", device_id=f"br-{participant_id}")
    p.central_job_id = job["job_id"]
    db.commit()
    return {"participant_id": participant_id, "job_id": job["job_id"]}


@app.get("/participants/{participant_id}/status")
def participant_status(participant_id: int, db: Session = Depends(get_db)):
    p = db.query(Participant).filter(Participant.id == participant_id).first()
    if not p or not p.central_job_id:
        return {"error": "No job"}
    job = client.get_job(p.central_job_id)
    if job.get("result_id"):
        result = client.get_job_result(p.central_job_id)
        p.result_id = job["result_id"]
        p.profile_json = json.dumps(result.get("profile") or {})
        db.commit()
        return {"job": job, "result": result}
    return {"job": job}


@app.get("/team/report")
def team_report(db: Session = Depends(get_db)):
    people = db.query(Participant).filter(Participant.profile_json.isnot(None)).all()
    strengths = []
    for p in people:
        profile = json.loads(p.profile_json or "{}")
        if profile.get("standout_strength"):
            strengths.append({"name": p.name, "strength": profile["standout_strength"]})
    return {
        "participants": len(people),
        "individual_strengths": strengths,
        "team_patterns": ["Diverse standout strengths" if len(set(s["strength"] for s in strengths)) > 1 else "Aligned strengths"],
        "gaps": ["Schedule follow-up for incomplete profiles"] if len(people) < 2 else [],
    }


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <html><body style="font-family:sans-serif;max-width:720px;margin:2rem auto">
      <h1>Boardroom Breakthroughs™</h1>
      <p>Create participant, upload handwriting, then view team report.</p>
      <p>Use API docs at <a href="/docs">/docs</a>.</p>
    </body></html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.boardroom.app.main:app", host="0.0.0.0", port=8002, reload=True)

import os
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import uvicorn
import asyncio
import json

from . import models, schemas, database, dependencies
from .services import image_processing, assessment

# Create tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Handwritten Notes OCR API")

# Use absolute path to ensure the public directory is always found
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
public_dir = os.path.join(BASE_DIR, "public")
app.mount("/public", StaticFiles(directory=public_dir), name="public")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Handwritten Notes OCR API"}

@app.post("/notes/analyze/", response_model=schemas.AnalyzeResponse)
async def analyze_note(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    
    # Process image for quality only
    result = image_processing.analyze_image_quality(image_bytes)
    
    return schemas.AnalyzeResponse(
        scan_quality=result["scan_quality"],
        handwriting_detected=result["handwriting_detected"],
        retake=result["retake"]
    )

@app.post("/notes/advanced_analysis/", response_model=schemas.AdvancedAnalysisResponse)
async def advanced_analysis_note(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    
    # Process image for advanced characteristics
    result = image_processing.perform_advanced_analysis(image_bytes)
    
    return schemas.AdvancedAnalysisResponse(
        slant=result["slant"],
        spacing=result["spacing"],
        shapes=result["shapes"],
        relative_size=result["relative_size"],
        consistency=result["consistency"],
        stroke_geometry=result["stroke_geometry"]
    )


@app.post("/notes/upload/", response_model=schemas.UploadResponse)
async def upload_note(
    file: UploadFile = File(...),
    device: models.Device = Depends(dependencies.get_current_device),
    db: Session = Depends(database.get_db)
):
    # Read image file
    image_bytes = await file.read()
    
    # Process image
    result = image_processing.process_handwritten_note(image_bytes)
    
    # Generate profile for DB
    profile = assessment.generate_profile(result.get("boxes", []), result["detected_text"])

    # Save to database
    upload_record = models.NoteUpload(
        device_id=device.id,
        filename=file.filename,
        scan_quality=result["scan_quality"],
        handwriting_detected=result["handwriting_detected"],
        retake=result["retake"],
        detected_text=result["detected_text"],
        profile_data=json.dumps(profile)
    )
    db.add(upload_record)
    db.commit()
    db.refresh(upload_record)
    
    return schemas.UploadResponse(
        scan_quality=result["scan_quality"],
        handwriting_detected=result["handwriting_detected"],
        retake=result["retake"],
        detected_text=result["detected_text"],
        profile=profile
    )

@app.post("/notes/upload_stream/")
async def upload_note_stream(
    file: UploadFile = File(...),
    device: models.Device = Depends(dependencies.get_current_device),
    db: Session = Depends(database.get_db)
):
    async def event_generator():
        image_bytes = await file.read()
        
        # 1. Preparing document
        yield f"data: {json.dumps({'status': 'Preparing document...'})}\n\n"
        await asyncio.sleep(1)
        
        # 2. Identifying characteristics
        yield f"data: {json.dumps({'status': 'Identifying handwriting characteristics...'})}\n\n"
        await asyncio.sleep(1)
        
        # Actually run the heavy OCR
        # For a production app this needs a dedicated task queue (like Celery).
        # For the demo, we run it directly in the event loop thread to prevent PaddleOCR C++ thread-safety crashes!
        result = image_processing.process_handwritten_note(image_bytes)
        
        # 3. Building your profile
        yield f"data: {json.dumps({'status': 'Building your profile...'})}\n\n"
        profile = assessment.generate_profile(result.get("boxes", []), result.get("detected_text", ""))
        await asyncio.sleep(1)
        
        # 4. Preparing personalised insights
        yield f"data: {json.dumps({'status': 'Preparing personalised insights...'})}\n\n"
        await asyncio.sleep(1)
        
        # Save to database
        upload_record = models.NoteUpload(
            device_id=device.id,
            filename=file.filename,
            scan_quality=result["scan_quality"],
            handwriting_detected=result["handwriting_detected"],
            retake=result["retake"],
            detected_text=result["detected_text"],
            profile_data=json.dumps(profile)
        )
        db.add(upload_record)
        db.commit()
        db.refresh(upload_record)
        
        # Final result
        final_response = schemas.UploadResponse(
            scan_quality=result["scan_quality"],
            handwriting_detected=result["handwriting_detected"],
            retake=result["retake"],
            detected_text=result["detected_text"],
            profile=profile
        )
        
        yield f"data: {json.dumps({'status': 'Complete', 'result': final_response.dict()})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/journey/", response_model=schemas.JourneyResponse)
def get_journey(
    device: models.Device = Depends(dependencies.get_current_device),
    db: Session = Depends(database.get_db)
):
    # Fetch the latest upload for this device to determine the Current Focus
    latest_upload = db.query(models.NoteUpload).filter(
        models.NoteUpload.device_id == device.id
    ).order_by(models.NoteUpload.created_at.desc()).first()

    current_focus = "Adaptive Communication" # Fallback default
    weeks_complete = 0
    
    if latest_upload:
        if latest_upload.profile_data:
            try:
                profile = json.loads(latest_upload.profile_data)
                current_focus = profile.get("standout_strength", current_focus)
            except json.JSONDecodeError:
                pass
                
        # Calculate weeks passed since the upload
        delta = datetime.utcnow() - latest_upload.created_at
        weeks_complete = min(delta.days // 7, 4)

    # Build the 4-week curriculum timeline
    weeks = [
        schemas.JourneyWeek(
            week_number=1,
            title="Self-awareness",
            description="Understand your natural patterns and tendencies.",
            status="completed" if weeks_complete >= 1 else ("active" if weeks_complete == 0 else "locked")
        ),
        schemas.JourneyWeek(
            week_number=2,
            title="Communication",
            description="Build awareness of how you express ideas and connect with others.",
            status="completed" if weeks_complete >= 2 else ("active" if weeks_complete == 1 else "locked")
        ),
        schemas.JourneyWeek(
            week_number=3,
            title="Adaptability",
            description="Explore how you respond to change, uncertainty, and new situations.",
            status="completed" if weeks_complete >= 3 else ("active" if weeks_complete == 2 else "locked")
        ),
        schemas.JourneyWeek(
            week_number=4,
            title="Applied Practice",
            description="Put your Insights into action in everyday situations.",
            status="completed" if weeks_complete >= 4 else ("active" if weeks_complete == 3 else "locked")
        )
    ]

    return schemas.JourneyResponse(
        current_focus=current_focus,
        weeks_complete=weeks_complete,
        total_weeks=4,
        weeks=weeks
    )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

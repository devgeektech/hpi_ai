from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os

from .database import engine, Base
from .routers import auth, jobs, images, handwriting, hpi_routes, reviews, stubs

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Central AI Platform", version="1.0.0")

PREFIX = "/api/v1"
app.include_router(auth.router, prefix=PREFIX)
app.include_router(jobs.router, prefix=PREFIX)
app.include_router(images.router, prefix=PREFIX)
app.include_router(handwriting.router, prefix=PREFIX)
app.include_router(hpi_routes.router, prefix=PREFIX)
app.include_router(reviews.router, prefix=PREFIX)
app.include_router(stubs.router, prefix=PREFIX)


@app.get(f"{PREFIX}/health")
def health():
    return {"status": "ok", "service": "central-ai"}


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
hpi_public = os.path.join(BASE_DIR, "apps", "hpi", "public")
if os.path.isdir(hpi_public):
    app.mount("/public", StaticFiles(directory=hpi_public), name="public")


@app.get("/")
def root():
    return {
        "message": "Central AI Platform",
        "docs": "/docs",
        "api": "/api/v1/health",
        "ui": "/public/index.html",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.central_ai.app.main:app", host="0.0.0.0", port=8000, reload=True)

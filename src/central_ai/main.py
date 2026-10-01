"""Central AI FastAPI Application Entry Point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import health_router, images_router, handwriting_router

app = FastAPI(
    title="Central AI — Handwriting Analysis",
    version="1.0.0",
    description=(
        "Stateless Handwriting OCR and HPI Assessment API. "
        "Supports multi-tenant integration across consumer applications via X-Project-Id."
    ),
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API v1 Routers
API_V1_PREFIX = "/api/v1"
app.include_router(health_router, prefix=API_V1_PREFIX)
app.include_router(images_router, prefix=API_V1_PREFIX)
app.include_router(handwriting_router, prefix=API_V1_PREFIX)


@app.get("/", tags=["root"])
def root():
    """Service status and endpoint index."""
    return {
        "service": "Central AI — Handwriting Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": f"{API_V1_PREFIX}/health",
        "analyze": f"{API_V1_PREFIX}/handwriting/analyze",
        "validate": f"{API_V1_PREFIX}/images/validate",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("central_ai.main:app", host="127.0.0.1", port=8000, reload=False)

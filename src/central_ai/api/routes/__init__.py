"""API route controllers."""

from .health import router as health_router
from .images import router as images_router
from .handwriting import router as handwriting_router

__all__ = ["health_router", "images_router", "handwriting_router"]

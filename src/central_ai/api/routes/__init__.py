"""API route controllers."""

from .health import router as health_router
from .handwriting import router as handwriting_router
from .hpi import router as hpi_router
from .training import router as training_router

__all__ = ["health_router", "handwriting_router", "hpi_router", "training_router"]


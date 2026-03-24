"""
Main FastAPI application entry point.
"""

import logging
import os

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import init_model, router
from app.config import settings
from app.utils.database import init_db

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated on_event handlers)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown logic."""
    # --- Startup ---
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("Device: %s", settings.DEVICE)

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    init_db()

    try:
        init_model()
        logger.info("Model initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize model on startup: %s", e, exc_info=True)

    yield  # Application runs here

    # --- Shutdown ---
    logger.info("Shutting down %s.", settings.APP_NAME)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered deepfake detection API",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Root health-check endpoint."""
    return {
        "message": "Deepfake Detector API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

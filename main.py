from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.endpoints import health, recommendations, merchants, orders, forecasting, fraud, metrics
from app.db.session import engine
from app.db.base import Base
# Ensure models are imported so metadata includes their tables
from app.db import models  # noqa: F401

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the FastAPI application."""
    # Startup
    logger.info("Starting up Wasaa Storefront application...")
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Wasaa Storefront application...")

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Wasaa Storefront API - ML-powered e-commerce platform",
    version="1.0.0",
    debug=settings.DEBUG,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Include routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(recommendations.router, prefix="/api/v1")
app.include_router(merchants.router, prefix="/api/v1")
app.include_router(orders.router, prefix="/api/v1")
app.include_router(forecasting.router, prefix="/api/v1")
app.include_router(fraud.router, prefix="/api/v1")
app.include_router(metrics.router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Wasaa Storefront API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
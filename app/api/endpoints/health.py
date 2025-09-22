from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import logging
from app.db.session import SessionLocal
from sqlalchemy.exc import SQLAlchemyError

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Comprehensive health check endpoint.
    Checks:
    - API is running
    - Database connectivity
    - Optional: downstream services (AdsManager, ML services, etc.)
    """
    health_status = {
        "api": "ok",
        "database": "unknown",
        "services": {}
    }

    # Check database connectivity
    try:
        with SessionLocal() as db:
            db.execute("SELECT 1")
        health_status["database"] = "ok"
    except SQLAlchemyError as e:
        logging.error(f"Database health check failed: {e}")
        health_status["database"] = "failed"

    # Optional: Add checks for external services
    try:
        # Example: AdsManagerService.ping() if implemented
        from app.services.ads_manager_service import AdsManagerService
        if hasattr(AdsManagerService, "ping"):
            health_status["services"]["ads_manager"] = "ok" if AdsManagerService.ping() else "failed"
    except Exception as e:
        logging.error(f"AdsManager service check failed: {e}")
        health_status["services"]["ads_manager"] = "failed"

    # Overall status
    overall_status = status.HTTP_200_OK if all(
        v == "ok" for v in [health_status["api"], health_status["database"]] + list(health_status["services"].values())
    ) else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(status_code=overall_status, content=health_status)

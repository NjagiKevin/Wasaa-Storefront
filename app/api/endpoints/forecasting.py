from fastapi import APIRouter
from app.services.demand_forecast_service import DemandForecastService

router = APIRouter(prefix="/forecasting", tags=["Forecasting"])

@router.post("/demand")
def demand_forecast(payload: dict):
    product_ids = payload.get("product_ids", [])
    return DemandForecastService.predict_demand(product_ids)
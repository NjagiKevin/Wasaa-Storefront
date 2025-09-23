from fastapi import APIRouter
from app.services.fraud_service import FraudService

router = APIRouter(prefix="/fraud", tags=["Fraud"])

@router.post("/score")
def score_event(event: dict):
    return {"score": FraudService.score_event(event)}
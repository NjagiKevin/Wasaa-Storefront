from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import time

from app.db import crud
from app.api.dependencies import get_db
from app.utils.cache import cache_get, cache_set
from app.utils.metrics import recommendation_requests_total, recommendation_request_latency_seconds

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])

@router.get("/products/{user_id}")
def recommend_products(user_id: str, db: Session = Depends(get_db)):
    """
    Return recommended products for a given user.
    Uses Redis cache if available; falls back to DB logic.
    """
    endpoint = "/recommendations/products/{user_id}"
    recommendation_requests_total.labels(endpoint=endpoint).inc()
    t0 = time.perf_counter()

    cache_key = f"reco:{user_id}"
    cached = cache_get(cache_key)
    if cached:
        recommendation_request_latency_seconds.labels(endpoint=endpoint).observe(time.perf_counter() - t0)
        return cached

    products = crud.get_recommended_products(db=db, user_id=user_id)
    payload = {
        "user_id": user_id,
        "recommended_products": products,
        "count": len(products)
    }
    cache_set(cache_key, payload, ttl_seconds=600)
    recommendation_request_latency_seconds.labels(endpoint=endpoint).observe(time.perf_counter() - t0)
    return payload

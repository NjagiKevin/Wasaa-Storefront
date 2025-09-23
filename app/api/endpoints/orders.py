from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db

router = APIRouter(prefix="/orders", tags=["Orders"])

@router.get("/")
def list_orders(db: Session = Depends(get_db)):
    """List all orders - mock implementation"""
    return {
        "message": "Orders endpoint",
        "orders": [
            {"id": "order-1", "status": "completed", "total": 99.99},
            {"id": "order-2", "status": "pending", "total": 149.50}
        ]
    }

@router.get("/{order_id}")
def get_order(order_id: str, db: Session = Depends(get_db)):
    """Get order by ID - mock implementation"""
    return {
        "id": order_id,
        "status": "completed",
        "total": 99.99,
        "items": [{"product_id": "prod-1", "quantity": 2}],
        "created_at": "2024-01-01T00:00:00Z"
    }

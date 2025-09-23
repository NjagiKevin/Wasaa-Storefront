from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.api.dependencies import get_db

router = APIRouter(prefix="/merchants", tags=["Merchants"])

@router.get("/")
def list_merchants(db: Session = Depends(get_db)):
    """List all merchants - mock implementation"""
    return {
        "message": "Merchants endpoint",
        "merchants": [
            {"id": "merchant-1", "name": "Store 1", "status": "active"},
            {"id": "merchant-2", "name": "Store 2", "status": "active"}
        ]
    }

@router.get("/{merchant_id}")
def get_merchant(merchant_id: str, db: Session = Depends(get_db)):
    """Get merchant by ID - mock implementation"""
    return {
        "id": merchant_id,
        "name": f"Merchant {merchant_id}",
        "status": "active",
        "created_at": "2024-01-01T00:00:00Z"
    }

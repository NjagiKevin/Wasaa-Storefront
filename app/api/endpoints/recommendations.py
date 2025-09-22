from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from app.db import crud
from app.db.schemas import ProductRead
from app.api.dependencies import get_db

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])

@router.get("/products/{user_id}", response_model=List[ProductRead])
def recommend_products(user_id: str, db: Session = Depends(get_db)):
    """
    Return recommended products for a given user.
    For demo purposes, returns top products or products based on simple logic.
    """
    return crud.get_recommended_products(db=db, user_id=user_id)

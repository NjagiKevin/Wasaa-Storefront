from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.db import crud
from app.db.schemas import OrderCreate, OrderRead
from app.api.dependencies import get_db

router = APIRouter(prefix="/orders", tags=["Orders"])

@router.post("/", response_model=OrderRead, status_code=status.HTTP_201_CREATED)
def create_order(order_in: OrderCreate, db: Session = Depends(get_db)):
    return crud.create_order(db=db, order_in=order_in)

@router.get("/", response_model=List[OrderRead])
def list_orders(db: Session = Depends(get_db)):
    return crud.get_orders(db=db)

@router.get("/{order_id}", response_model=OrderRead)
def get_order(order_id: str, db: Session = Depends(get_db)):
    order = crud.get_order(db=db, order_id=order_id)
    if not order:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")
    return order

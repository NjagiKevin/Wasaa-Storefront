from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.models import Order
from app.schemas.order_schema import OrderCreate, OrderUpdate

def get_orders(db: Session, skip: int = 0, limit: int = 100) -> List[Order]:
    return db.query(Order).offset(skip).limit(limit).all()

def get_order_by_id(db: Session, order_id: str) -> Optional[Order]:
    return db.query(Order).filter(Order.order_id == order_id).first()

def create_order(db: Session, order: OrderCreate) -> Order:
    db_order = Order(**order.dict())
    db.add(db_order)
    db.commit()
    db.refresh(db_order)
    return db_order

def update_order(db: Session, db_order: Order, updates: OrderUpdate) -> Order:
    for field, value in updates.dict(exclude_unset=True).items():
        setattr(db_order, field, value)
    db.commit()
    db.refresh(db_order)
    return db_order

def delete_order(db: Session, db_order: Order) -> None:
    db.delete(db_order)
    db.commit()

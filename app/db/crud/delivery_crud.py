from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.models import Delivery
from app.schemas.delivery_schema import DeliveryCreate, DeliveryUpdate

def get_deliveries(db: Session, skip: int = 0, limit: int = 100) -> List[Delivery]:
    return db.query(Delivery).offset(skip).limit(limit).all()

def get_delivery_by_id(db: Session, delivery_id: str) -> Optional[Delivery]:
    return db.query(Delivery).filter(Delivery.delivery_id == delivery_id).first()

def create_delivery(db: Session, delivery: DeliveryCreate) -> Delivery:
    db_delivery = Delivery(**delivery.dict())
    db.add(db_delivery)
    db.commit()
    db.refresh(db_delivery)
    return db_delivery

def update_delivery(db: Session, db_delivery: Delivery, updates: DeliveryUpdate) -> Delivery:
    for field, value in updates.dict(exclude_unset=True).items():
        setattr(db_delivery, field, value)
    db.commit()
    db.refresh(db_delivery)
    return db_delivery

def delete_delivery(db: Session, db_delivery: Delivery) -> None:
    db.delete(db_delivery)
    db.commit()

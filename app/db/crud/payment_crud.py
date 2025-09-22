from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.models import Payment
from app.schemas.payment_schema import PaymentCreate, PaymentUpdate

def get_payments(db: Session, skip: int = 0, limit: int = 100) -> List[Payment]:
    return db.query(Payment).offset(skip).limit(limit).all()

def get_payment_by_id(db: Session, payment_id: str) -> Optional[Payment]:
    return db.query(Payment).filter(Payment.payment_id == payment_id).first()

def create_payment(db: Session, payment: PaymentCreate) -> Payment:
    db_payment = Payment(**payment.dict())
    db.add(db_payment)
    db.commit()
    db.refresh(db_payment)
    return db_payment

def update_payment(db: Session, db_payment: Payment, updates: PaymentUpdate) -> Payment:
    for field, value in updates.dict(exclude_unset=True).items():
        setattr(db_payment, field, value)
    db.commit()
    db.refresh(db_payment)
    return db_payment

def delete_payment(db: Session, db_payment: Payment) -> None:
    db.delete(db_payment)
    db.commit()

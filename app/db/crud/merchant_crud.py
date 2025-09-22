from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.models import Merchant
from app.schemas.merchant_schema import MerchantCreate, MerchantUpdate

def get_merchants(db: Session, skip: int = 0, limit: int = 100) -> List[Merchant]:
    return db.query(Merchant).offset(skip).limit(limit).all()

def get_merchant_by_id(db: Session, merchant_id: str) -> Optional[Merchant]:
    return db.query(Merchant).filter(Merchant.merchant_id == merchant_id).first()

def create_merchant(db: Session, merchant: MerchantCreate) -> Merchant:
    db_merchant = Merchant(**merchant.dict())
    db.add(db_merchant)
    db.commit()
    db.refresh(db_merchant)
    return db_merchant

def update_merchant(db: Session, db_merchant: Merchant, updates: MerchantUpdate) -> Merchant:
    for field, value in updates.dict(exclude_unset=True).items():
        setattr(db_merchant, field, value)
    db.commit()
    db.refresh(db_merchant)
    return db_merchant

def delete_merchant(db: Session, db_merchant: Merchant) -> None:
    db.delete(db_merchant)
    db.commit()

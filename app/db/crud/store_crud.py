from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.models import Store
from app.schemas.store_schema import StoreCreate, StoreUpdate

def get_stores(db: Session, skip: int = 0, limit: int = 100) -> List[Store]:
    return db.query(Store).offset(skip).limit(limit).all()

def get_store_by_id(db: Session, store_id: str) -> Optional[Store]:
    return db.query(Store).filter(Store.store_id == store_id).first()

def create_store(db: Session, store: StoreCreate) -> Store:
    db_store = Store(**store.dict())
    db.add(db_store)
    db.commit()
    db.refresh(db_store)
    return db_store

def update_store(db: Session, db_store: Store, updates: StoreUpdate) -> Store:
    for field, value in updates.dict(exclude_unset=True).items():
        setattr(db_store, field, value)
    db.commit()
    db.refresh(db_store)
    return db_store

def delete_store(db: Session, db_store: Store) -> None:
    db.delete(db_store)
    db.commit()

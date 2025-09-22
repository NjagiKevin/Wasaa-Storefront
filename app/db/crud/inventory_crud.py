from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.models import Inventory
from app.schemas.inventory_schema import InventoryCreate, InventoryUpdate

def get_inventory(db: Session, skip: int = 0, limit: int = 100) -> List[Inventory]:
    return db.query(Inventory).offset(skip).limit(limit).all()

def get_inventory_by_id(db: Session, inventory_id: str) -> Optional[Inventory]:
    return db.query(Inventory).filter(Inventory.inventory_id == inventory_id).first()

def create_inventory(db: Session, inventory: InventoryCreate) -> Inventory:
    db_inventory = Inventory(**inventory.dict())
    db.add(db_inventory)
    db.commit()
    db.refresh(db_inventory)
    return db_inventory

def update_inventory(db: Session, db_inventory: Inventory, updates: InventoryUpdate) -> Inventory:
    for field, value in updates.dict(exclude_unset=True).items():
        setattr(db_inventory, field, value)
    db.commit()
    db.refresh(db_inventory)
    return db_inventory

def delete_inventory(db: Session, db_inventory: Inventory) -> None:
    db.delete(db_inventory)
    db.commit()

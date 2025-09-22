from uuid import UUID
from datetime import datetime
from pydantic import BaseModel

class InventoryBase(BaseModel):
    product_id: UUID
    stock_quantity: int = 0
    reserved_quantity: int = 0

class InventoryCreate(InventoryBase):
    pass

class InventoryUpdate(BaseModel):
    stock_quantity: int | None
    reserved_quantity: int | None

class InventoryRead(InventoryBase):
    inventory_id: UUID
    updated_at: datetime

    class Config:
        orm_mode = True

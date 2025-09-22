from uuid import UUID
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

class ProductStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    deleted = "deleted"

class ProductBase(BaseModel):
    store_id: UUID
    category_id: UUID
    name: str
    description: str | None = None
    price: float
    currency: str = "KES"
    status: ProductStatus = ProductStatus.active

class ProductCreate(ProductBase):
    pass

class ProductUpdate(BaseModel):
    name: str | None
    description: str | None
    price: float | None
    currency: str | None
    status: ProductStatus | None

class ProductRead(ProductBase):
    product_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

from uuid import UUID
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

class StoreStatus(str, Enum):
    active = "active"
    suspended = "suspended"
    closed = "closed"

class StoreBase(BaseModel):
    merchant_id: UUID
    name: str
    description: str | None = None
    logo_url: str | None = None
    status: StoreStatus = StoreStatus.active

class StoreCreate(StoreBase):
    pass

class StoreUpdate(BaseModel):
    name: str | None
    description: str | None
    logo_url: str | None
    status: StoreStatus | None

class StoreRead(StoreBase):
    store_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

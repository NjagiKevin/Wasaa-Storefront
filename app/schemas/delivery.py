from uuid import UUID
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

class DeliveryStatus(str, Enum):
    pending = "pending"
    dispatched = "dispatched"
    in_transit = "in_transit"
    delivered = "delivered"
    failed = "failed"

class DeliveryBase(BaseModel):
    order_id: UUID
    courier_id: UUID | None = None
    status: DeliveryStatus = DeliveryStatus.pending
    dispatched_at: datetime | None = None
    delivered_at: datetime | None = None

class DeliveryCreate(DeliveryBase):
    pass

class DeliveryUpdate(BaseModel):
    status: DeliveryStatus | None
    dispatched_at: datetime | None
    delivered_at: datetime | None

class DeliveryRead(DeliveryBase):
    delivery_id: UUID

    class Config:
        orm_mode = True

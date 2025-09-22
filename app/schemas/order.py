from typing import List, Optional
from uuid import UUID
import datetime
from pydantic import BaseModel
from enum import Enum

class OrderStatus(str, Enum):
    pending = "pending"
    confirmed = "confirmed"
    shipped = "shipped"
    delivered = "delivered"
    refunded = "refunded"
    cancelled = "cancelled"

class OrderItemSchema(BaseModel):
    product_id: UUID
    quantity: int
    price: float

class OrderBase(BaseModel):
    customer_id: UUID
    store_id: UUID
    total_amount: float
    currency: str = "KES"
    escrow_id: UUID

class OrderCreate(OrderBase):
    order_items: List[OrderItemSchema]

class OrderUpdate(BaseModel):
    status: Optional[OrderStatus]

class OrderRead(OrderBase):
    order_id: UUID
    status: OrderStatus
    created_at: datetime.datetime
    updated_at: datetime.datetime
    order_items: Optional[List[OrderItemSchema]] = []

    class Config:
        orm_mode = True

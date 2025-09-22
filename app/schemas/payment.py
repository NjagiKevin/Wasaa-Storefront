from uuid import UUID
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

class PaymentMethod(str, Enum):
    wallet = "wallet"
    card = "card"
    mobile_money = "mobile_money"
    bank_transfer = "bank_transfer"

class PaymentStatus(str, Enum):
    pending = "pending"
    completed = "completed"
    failed = "failed"
    refunded = "refunded"

class PaymentBase(BaseModel):
    order_id: UUID
    wallet_id: UUID
    method: PaymentMethod
    amount: float
    transaction_ref: str

class PaymentCreate(PaymentBase):
    pass

class PaymentUpdate(BaseModel):
    status: PaymentStatus | None

class PaymentRead(PaymentBase):
    payment_id: UUID
    status: PaymentStatus
    created_at: datetime

    class Config:
        orm_mode = True

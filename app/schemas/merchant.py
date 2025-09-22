from uuid import UUID
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

class KYCStatus(str, Enum):
    pending = "pending"
    verified = "verified"
    rejected = "rejected"

class MerchantBase(BaseModel):
    user_id: UUID
    wallet_id: UUID
    kyc_status: KYCStatus = KYCStatus.pending

class MerchantCreate(MerchantBase):
    pass

class MerchantUpdate(BaseModel):
    kyc_status: KYCStatus

class MerchantRead(MerchantBase):
    merchant_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

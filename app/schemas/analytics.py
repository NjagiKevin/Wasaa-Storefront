from uuid import UUID
from datetime import datetime
from pydantic import BaseModel

class AnalyticsBase(BaseModel):
    merchant_id: UUID
    period: str
    total_sales: float = 0
    total_orders: int = 0

class AnalyticsCreate(AnalyticsBase):
    pass

class AnalyticsRead(AnalyticsBase):
    report_id: UUID
    generated_at: datetime

    class Config:
        orm_mode = True

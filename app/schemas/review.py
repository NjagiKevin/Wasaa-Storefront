from uuid import UUID
from datetime import datetime
from pydantic import BaseModel

class ReviewBase(BaseModel):
    order_id: UUID
    product_id: UUID
    customer_id: UUID
    rating: int
    comment: str | None = None

class ReviewCreate(ReviewBase):
    pass

class ReviewUpdate(BaseModel):
    rating: int | None
    comment: str | None

class ReviewRead(ReviewBase):
    review_id: UUID
    created_at: datetime

    class Config:
        orm_mode = True

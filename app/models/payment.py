# app/models/payment.py
import datetime
from sqlalchemy import Column, DECIMAL, TIMESTAMP, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.db.base import Base

class Payment(Base):
    __tablename__ = "payments"

    payment_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False, unique=True)
    order_id = Column(UUID(as_uuid=True), nullable=False)  # removed FK
    wallet_id = Column(UUID(as_uuid=True), nullable=False)  # removed FK
    amount = Column(DECIMAL(12,2), nullable=False)
    currency = Column(String(10), nullable=False, default="KES")
    status = Column(String(20), nullable=False, default="pending")
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.datetime.utcnow)

   

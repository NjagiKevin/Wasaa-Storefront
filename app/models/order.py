import enum
import datetime
from sqlalchemy import Column, Enum, TIMESTAMP, DECIMAL, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.db.base import Base

class OrderStatus(enum.Enum):
    pending = "pending"
    confirmed = "confirmed"
    shipped = "shipped"
    delivered = "delivered"
    refunded = "refunded"
    cancelled = "cancelled"

class Order(Base):
    __tablename__ = "orders"

    order_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False, unique=True)
    customer_id = Column(UUID(as_uuid=True), nullable=False)  # Removed FK
    store_id = Column(UUID(as_uuid=True), nullable=False)  # Removed FK
    total_amount = Column(DECIMAL(12,2), nullable=False)
    currency = Column(String(10), nullable=False, default="KES")
    status = Column(Enum(OrderStatus), nullable=False, default=OrderStatus.pending)
    escrow_id = Column(UUID(as_uuid=True), nullable=False)  # Removed FK
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(TIMESTAMP, nullable=False, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    
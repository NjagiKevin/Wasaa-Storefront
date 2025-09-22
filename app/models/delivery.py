import enum
import datetime
from sqlalchemy import Column, Enum, TIMESTAMP, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.db.base import Base

class DeliveryStatus(enum.Enum):
    pending = "pending"
    dispatched = "dispatched"
    in_transit = "in_transit"
    delivered = "delivered"
    failed = "failed"


class Delivery(Base):
    __tablename__ = "deliveries"

    delivery_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False, unique=True)
    order_id = Column(UUID(as_uuid=True), nullable=False)  # removed FK
    courier_id = Column(UUID(as_uuid=True), nullable=False)  # removed FK
    status = Column(String(20), nullable=False, default="pending")
    tracking_code = Column(String(50), nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(TIMESTAMP, nullable=False, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


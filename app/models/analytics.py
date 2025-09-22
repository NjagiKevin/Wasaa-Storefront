import datetime
from sqlalchemy import Column, TIMESTAMP, Integer, DECIMAL, String
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base

class Analytics(Base):
    __tablename__ = "analytics"

    report_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False, unique=True)
    merchant_id = Column(UUID(as_uuid=True), nullable=False)  # Removed FK
    period = Column(String, nullable=False)
    total_sales = Column(DECIMAL(12,2), nullable=False, default=0)
    total_orders = Column(Integer, nullable=False, default=0)
    generated_at = Column(TIMESTAMP, nullable=False, default=datetime.datetime.utcnow)

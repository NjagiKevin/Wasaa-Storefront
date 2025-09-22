import datetime
from sqlalchemy import Column, String, DECIMAL, TIMESTAMP, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.db.base import Base

class Product(Base):
    __tablename__ = "products"

    product_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False, unique=True)
    description = Column(Text)
    store_id = Column(UUID(as_uuid=True), nullable=False)  # Removed FK
    category_id = Column(UUID(as_uuid=True), nullable=False)  # Removed FK
    name = Column(String, nullable=False)
    price = Column(DECIMAL(12,2), nullable=False)
    currency = Column(String(10), nullable=False, default="KES")
    status = Column(String(20), nullable=False, default="active")
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(TIMESTAMP, nullable=False, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    
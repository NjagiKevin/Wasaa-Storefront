import datetime
from sqlalchemy import Column, TIMESTAMP, String, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.db.base import Base

class Review(Base):
    __tablename__ = "reviews"

    review_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False, unique=True)
    order_id = Column(UUID(as_uuid=True), nullable=False)  # Removed FK
    product_id = Column(UUID(as_uuid=True), nullable=False)  # Removed FK
    customer_id = Column(UUID(as_uuid=True), nullable=False)  # Removed FK
    rating = Column(Integer, nullable=False)
    comment = Column(String, nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.datetime.utcnow)



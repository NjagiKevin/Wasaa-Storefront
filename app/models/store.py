import datetime
from sqlalchemy import Column, String, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.db.base import Base

class Store(Base):
    __tablename__ = "stores"

    store_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False, unique=True)
    merchant_id = Column(UUID(as_uuid=True), nullable=False)  # Removed FK
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    logo_url = Column(String, nullable=True)
    status = Column(String(20), nullable=False, default="active")
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(TIMESTAMP, nullable=False, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    
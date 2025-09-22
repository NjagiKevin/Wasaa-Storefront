import datetime
from sqlalchemy import Column, TIMESTAMP, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.db.base import Base

class Inventory(Base):
    __tablename__ = "inventory"

    inventory_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False, unique=True)
    product_id = Column(UUID(as_uuid=True), nullable=False)  # Removed FK
    stock_quantity = Column(Integer, nullable=False, default=0)
    reserved_quantity = Column(Integer, nullable=False, default=0)
    updated_at = Column(TIMESTAMP, nullable=False, default=datetime.datetime.utcnow)



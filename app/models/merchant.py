import enum
import datetime
from sqlalchemy import Column, Enum, TIMESTAMP, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.db.base import Base  # assumes Base is declared in db/base.py

class KYCStatus(enum.Enum):
    pending = "pending"
    verified = "verified"
    rejected = "rejected"


class Merchant(Base):
    __tablename__ = "merchants"

    merchant_id = Column(UUID(as_uuid=True), primary_key=True, nullable=False, unique=True)
    user_id = Column(UUID(as_uuid=True), nullable=True, unique=True)
    wallet_id = Column(UUID(as_uuid=True), nullable=True, unique=True)  # Removed FK
    kyc_status = Column(String(20), nullable=False, default="pending")
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(TIMESTAMP, nullable=False, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

 

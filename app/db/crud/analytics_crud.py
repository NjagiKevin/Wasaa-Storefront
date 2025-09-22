from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.models import Analytics
from app.schemas.analytics_schema import AnalyticsCreate, AnalyticsUpdate

def get_analytics(db: Session, skip: int = 0, limit: int = 100) -> List[Analytics]:
    return db.query(Analytics).offset(skip).limit(limit).all()

def get_analytics_by_id(db: Session, analytics_id: str) -> Optional[Analytics]:
    return db.query(Analytics).filter(Analytics.analytics_id == analytics_id).first()

def create_analytics(db: Session, analytics: AnalyticsCreate) -> Analytics:
    db_analytics = Analytics(**analytics.dict())
    db.add(db_analytics)
    db.commit()
    db.refresh(db_analytics)
    return db_analytics

def update_analytics(db: Session, db_analytics: Analytics, updates: AnalyticsUpdate) -> Analytics:
    for field, value in updates.dict(exclude_unset=True).items():
        setattr(db_analytics, field, value)
    db.commit()
    db.refresh(db_analytics)
    return db_analytics

def delete_analytics(db: Session, db_analytics: Analytics) -> None:
    db.delete(db_analytics)
    db.commit()

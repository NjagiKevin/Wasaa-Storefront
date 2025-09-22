from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.models import Review
from app.schemas.review_schema import ReviewCreate, ReviewUpdate

def get_reviews(db: Session, skip: int = 0, limit: int = 100) -> List[Review]:
    return db.query(Review).offset(skip).limit(limit).all()

def get_review_by_id(db: Session, review_id: str) -> Optional[Review]:
    return db.query(Review).filter(Review.review_id == review_id).first()

def create_review(db: Session, review: ReviewCreate) -> Review:
    db_review = Review(**review.dict())
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return db_review

def update_review(db: Session, db_review: Review, updates: ReviewUpdate) -> Review:
    for field, value in updates.dict(exclude_unset=True).items():
        setattr(db_review, field, value)
    db.commit()
    db.refresh(db_review)
    return db_review

def delete_review(db: Session, db_review: Review) -> None:
    db.delete(db_review)
    db.commit()

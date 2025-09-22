from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.models import Product
from app.schemas.product_schema import ProductCreate, ProductUpdate

def get_products(db: Session, skip: int = 0, limit: int = 100) -> List[Product]:
    return db.query(Product).offset(skip).limit(limit).all()

def get_product_by_id(db: Session, product_id: str) -> Optional[Product]:
    return db.query(Product).filter(Product.product_id == product_id).first()

def create_product(db: Session, product: ProductCreate) -> Product:
    db_product = Product(**product.dict())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product

def update_product(db: Session, db_product: Product, updates: ProductUpdate) -> Product:
    for field, value in updates.dict(exclude_unset=True).items():
        setattr(db_product, field, value)
    db.commit()
    db.refresh(db_product)
    return db_product

def delete_product(db: Session, db_product: Product) -> None:
    db.delete(db_product)
    db.commit()

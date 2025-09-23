from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base

class Product(Base):
    __tablename__ = 'products'
    product_id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    category = Column(String, nullable=True)
    brand = Column(String, nullable=True)
    price = Column(Float, nullable=True)

class Inventory(Base):
    __tablename__ = 'inventory'
    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String, ForeignKey('products.product_id'))
    stock_quantity = Column(Integer, nullable=False, default=0)
    product = relationship('Product')

class UserProductInteraction(Base):
    __tablename__ = 'user_product_interactions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    product_id = Column(String, ForeignKey('products.product_id'))
    interaction_score = Column(Float, nullable=False, default=0.0)
    product = relationship('Product')
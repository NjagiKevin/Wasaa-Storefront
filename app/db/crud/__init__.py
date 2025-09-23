from sqlalchemy.orm import Session
from typing import List
from app.schemas.product import ProductRead

def get_recommended_products(db: Session, user_id: str, limit: int = 10) -> List[dict]:
    """
    Get recommended products for a user.
    For now, this returns mock data. In production, this would:
    1. Query user preferences and history
    2. Use ML models for recommendations
    3. Return actual products from database
    """
    # Mock recommended products
    mock_products = [
        {
            "product_id": f"product-{i}",
            "store_id": f"store-{i % 3}",
            "category_id": f"category-{i % 5}",
            "name": f"Recommended Product {i}",
            "description": f"Great product for user {user_id}",
            "price": 10.99 + (i * 5.0),
            "currency": "KES",
            "status": "active",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
        for i in range(1, min(limit + 1, 6))  # Return max 5 products for demo
    ]
    
    return mock_products
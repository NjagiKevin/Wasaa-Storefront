import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.base import Base
from app.db.session import engine

# Import all models so they are registered with Base
from app.models import merchant, store, product, inventory, order, payment, delivery, review, analytics

def create_tables():
    """
    Creates all tables in the database based on SQLAlchemy models.
    """
    print("Starting table creation...")
    Base.metadata.create_all(bind=engine)
    print("All tables have been successfully created!")

if __name__ == "__main__":
    create_tables()

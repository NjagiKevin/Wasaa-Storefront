import sys
import os

# Make sure the app module is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.base import Base
from app.db.session import engine

# Import all models so they are registered with SQLAlchemy Base
from app.models import merchant, store, product, inventory, order, payment, delivery, review, analytics

def init_db():
    """Create all tables in the database"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("All tables created successfully!")

if __name__ == "__main__":
    init_db()

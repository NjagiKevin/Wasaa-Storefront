from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Load database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/storefront")

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False  # Set to True for SQL debugging
)

# Create a configured "Session" class
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Dependency function for FastAPI
def get_db():
    """Yield a database session for dependency injection"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Wasaa Storefront"
    DEBUG: bool = True

    # Security
    SECRET_KEY: str = "change_this_to_a_secure_random_key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    # Database
    DATABASE_URL: Optional[str] = Field(default="postgresql://postgres:postgres@localhost:5432/storefront", env="DATABASE_URL")
    
    # Redis
    REDIS_URL: Optional[str] = Field(default="redis://localhost:6380/0", env="REDIS_URL")
    
    # Logging
    LOG_LEVEL: Optional[str] = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields

settings = Settings()

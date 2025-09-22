from pydantic import BaseSettings, PostgresDsn, Field

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Wasaa Storefront"
    DEBUG: bool = True

    # Security
    SECRET_KEY: str = "change_this_to_a_secure_random_key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    # Database
    DATABASE_URL: PostgresDsn = Field(..., env="DATABASE_URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

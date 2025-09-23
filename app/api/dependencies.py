from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError, jwt
import logging

from app.db.session import SessionLocal
from app.core.config import settings
from app.core.security import verify_token

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency.
    Creates a new SQLAlchemy session for each request and closes it when done.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """
    Optional authentication dependency.
    Returns user info if valid token is provided, None otherwise.
    Useful for endpoints that work for both authenticated and anonymous users.
    """
    if not credentials:
        return None
    
    try:
        payload = jwt.decode(
            credentials.credentials, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return {"user_id": user_id, "payload": payload}
    except JWTError:
        return None

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Required authentication dependency.
    Raises HTTPException if no valid token is provided.
    """
    try:
        payload = jwt.decode(
            credentials.credentials, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {"user_id": user_id, "payload": payload}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_admin_user(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """
    Admin-only dependency.
    Requires valid authentication and admin privileges.
    """
    user_role = current_user.get("payload", {}).get("role")
    if user_role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required"
        )
    return current_user

# Common query parameters
class CommonQueryParams:
    def __init__(
        self,
        skip: int = 0,
        limit: int = 100,
        sort_by: Optional[str] = None,
        sort_desc: bool = False
    ):
        self.skip = skip
        self.limit = min(limit, 1000)  # Maximum limit of 1000
        self.sort_by = sort_by
        self.sort_desc = sort_desc

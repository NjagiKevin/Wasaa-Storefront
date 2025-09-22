from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.db import crud, models
from app.schemas import MerchantCreate, MerchantRead
from app.api.dependencies import get_db

router = APIRouter(prefix="/merchants", tags=["Merchants"])

@router.post("/", response_model=MerchantRead, status_code=status.HTTP_201_CREATED)
def create_merchant(merchant_in: MerchantCreate, db: Session = Depends(get_db)):
    merchant = crud.create_merchant(db=db, merchant_in=merchant_in)
    return merchant

@router.get("/", response_model=List[MerchantRead])
def list_merchants(db: Session = Depends(get_db)):
    return crud.get_merchants(db=db)

@router.get("/{merchant_id}", response_model=MerchantRead)
def get_merchant(merchant_id: str, db: Session = Depends(get_db)):
    merchant = crud.get_merchant(db=db, merchant_id=merchant_id)
    if not merchant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Merchant not found")
    return merchant

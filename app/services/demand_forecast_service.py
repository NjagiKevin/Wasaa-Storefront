from typing import List, Dict
import logging
from app.db.session import SessionLocal
from app.db.models import Product, Inventory
import pandas as pd

class DemandForecastService:
    """
    Predict demand scores for products based on historical sales or inventory trends.
    """

    @staticmethod
    def predict_demand(product_ids: List[str]) -> Dict[str, float]:
        demand_scores = {}
        with SessionLocal() as db:
            for pid in product_ids:
                product = db.query(Product).filter(Product.product_id == pid).first()
                if product:
                    inventory = db.query(Inventory).filter(Inventory.product_id == pid).first()
                    stock = inventory.stock_quantity if inventory else 0
                    # Basic heuristic + clamp between 0.1 and 1.0
                    score = max(0.1, min(1.0, 1.0 - stock / 100))
                    demand_scores[pid] = score
                else:
                    logging.warning(f"Product {pid} not found in DB")
                    demand_scores[pid] = 0.5
        return demand_scores

    @staticmethod
    def predict_demand_for_campaigns(campaigns: List[Dict]) -> List[Dict]:
        """
        Enrich campaigns with demand scores for each product.
        """
        enriched_campaigns = []
        product_ids = [c["product_id"] for c in campaigns]
        scores = DemandForecastService.predict_demand(product_ids)
        for campaign in campaigns:
            pid = campaign["product_id"]
            campaign["demand_score"] = scores.get(pid, 0.5)
            enriched_campaigns.append(campaign)
        return enriched_campaigns

    @staticmethod
    def get_latest_accuracy() -> float:
        """Stub latest demand accuracy metric (e.g., MAPE-based)."""
        return 0.90

    @staticmethod
    def check_data_drift() -> Dict[str, bool]:
        """Stub drift check: return structure used by monitoring DAG."""
        return {"drift_detected": False}

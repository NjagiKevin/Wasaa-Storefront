from typing import List, Dict
from app.db.session import SessionLocal
from app.db.models import Product
import logging

class PricingService:
    """
    Calculate price attractiveness scores for products and integrate
    with campaigns for ML pipelines.
    """

    @staticmethod
    def calculate_price_score(product_ids: List[str]) -> Dict[str, float]:
        """
        Return a dict mapping product_id to a price score (0.0-1.0).
        Lower price gets higher score; missing price defaults to 0.5.
        """
        price_scores = {}
        with SessionLocal() as db:
            for pid in product_ids:
                product = db.query(Product).filter(Product.product_id == pid).first()
                if product and product.price:
                    price_scores[pid] = max(0.1, min(1.0, 1.0 / product.price))
                else:
                    logging.warning(f"Product {pid} missing or no price, defaulting score to 0.5")
                    price_scores[pid] = 0.5
        return price_scores

    @staticmethod
    def enrich_campaigns_with_price_score(campaigns: List[Dict]) -> List[Dict]:
        """
        Attach price scores to campaigns based on their associated product_ids.
        """
        product_ids = [c.get("product_id") for c in campaigns if c.get("product_id")]
        scores = PricingService.calculate_price_score(product_ids)
        for c in campaigns:
            c["price_score"] = scores.get(c.get("product_id"), 0.5)
        logging.info("Enriched campaigns with price scores")
        return campaigns

    @staticmethod
    def get_latest_price_accuracy() -> float:
        """Stub pricing accuracy metric for monitoring."""
        return 0.88

    @staticmethod
    def evaluate_pricing_effectiveness() -> Dict[str, float]:
        """Stub effectiveness summary for dynamic pricing monitoring."""
        return {"lift": 0.05, "avg_margin": 0.20}

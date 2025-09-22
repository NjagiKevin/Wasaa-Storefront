from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.db.session import SessionLocal
from app.db.models import Product, UserProductInteraction
from app.services.demand_forecast_service import DemandForecastService
from app.services.pricing_service import PricingService
import logging

class RecommendationService:
    """
    Hybrid recommendation service combining collaborative filtering,
    content-based filtering, demand forecast, and price scores.
    """

    demand_service = DemandForecastService()
    pricing_service = PricingService()

    @staticmethod
    def _get_product_features(db) -> pd.DataFrame:
        products = db.query(Product).all()
        data = [{"product_id": str(p.product_id), "category": p.category or "", 
                 "price": p.price or 0, "brand": p.brand or ""} for p in products]
        return pd.DataFrame(data)

    @staticmethod
    def _collaborative_filtering_scores(db, user_id: str) -> Dict[str, float]:
        interactions = db.query(UserProductInteraction).all()
        if not interactions:
            return {}
        df = pd.DataFrame([{"user_id": str(i.user_id), "product_id": str(i.product_id), 
                            "interaction": i.interaction_score} for i in interactions])
        user_product_matrix = df.pivot_table(index='user_id', columns='product_id', values='interaction', fill_value=0)
        if user_id not in user_product_matrix.index:
            logging.warning(f"User {user_id} not found in interactions")
            return {pid: 0.5 for pid in user_product_matrix.columns}
        user_vector = user_product_matrix.loc[user_id].values.reshape(1, -1)
        similarities = cosine_similarity(user_vector, user_product_matrix)[0]
        return {pid: float(sim) for pid, sim in zip(user_product_matrix.columns, similarities)}

    @staticmethod
    def _content_based_scores(db) -> Dict[str, float]:
        df = RecommendationService._get_product_features(db)
        if df.empty:
            return {}
        feature_matrix = pd.get_dummies(df[['category', 'brand']])
        similarity_matrix = cosine_similarity(feature_matrix)
        return {pid: float(np.mean(similarity_matrix[i])) for i, pid in enumerate(df['product_id'])}

    @staticmethod
    def recommend_products_for_campaigns(campaigns: List[Dict], top_n: int = 10) -> List[Dict]:
        """
        Enrich campaign/product data with hybrid recommendation scores.
        """
        enriched = []
        with SessionLocal() as db:
            product_ids = [str(c["product_id"]) for c in campaigns]
            collab_scores = {pid: 0.5 for pid in product_ids}  # placeholder, user-specific scores may not apply
            content_scores = RecommendationService._content_based_scores(db)
            demand_scores = RecommendationService.demand_service.predict_demand(product_ids)
            price_scores = RecommendationService.pricing_service.calculate_price_score(product_ids)

            for campaign in campaigns:
                pid = campaign["product_id"]
                campaign["recommendation_score"] = (
                    0.4 * collab_scores.get(pid, 0.5) +
                    0.3 * content_scores.get(pid, 0.5) +
                    0.2 * demand_scores.get(pid, 0.5) +
                    0.1 * price_scores.get(pid, 0.5)
                )
                enriched.append(campaign)

        return enriched

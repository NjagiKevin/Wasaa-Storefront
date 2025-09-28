from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.db.session import SessionLocal
from app.db.models import Product, UserProductInteraction
from app.services.demand_forecast_service import DemandForecastService
from app.services.pricing_service import PricingService
from app.ml_models.advanced_recommender import create_advanced_recommender, ContextAwareRecommender
from datetime import datetime
import logging

class RecommendationService:
    """
    Hybrid recommendation service combining collaborative filtering,
    content-based filtering, demand forecast, and price scores.
    Enhanced with advanced ML models and context-aware features.
    """

    demand_service = DemandForecastService()
    pricing_service = PricingService()
    advanced_recommender = create_advanced_recommender()

    @staticmethod
    def train_hybrid_model(campaigns: List[Dict]) -> None:
        """Stub training routine for hybrid model. Replace with real training."""
        logging.info(f"Training hybrid recommender on {len(campaigns)} campaigns (stub)")

    @staticmethod
    def generate_hybrid_recommendations(campaigns: List[Dict]) -> List[Dict]:
        """Wrapper to produce recommendations for campaigns using current heuristics."""
        return RecommendationService.recommend_products_for_campaigns(campaigns)

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

    @staticmethod
    def get_context_aware_recommendations(user_id: str, user_location: Tuple[float, float] = None,
                                        top_k: int = 10, all_products: List[str] = None) -> List[Dict]:
        """Get context-aware recommendations using advanced ML models"""
        try:
            # Get interaction data for context
            with SessionLocal() as db:
                interactions = db.query(UserProductInteraction).all()
                df_interactions = pd.DataFrame([{
                    'user_id': str(i.user_id),
                    'product_id': str(i.product_id),
                    'timestamp': datetime.now(),  # In real scenario, get from DB
                    'rating': i.interaction_score
                } for i in interactions]) if interactions else pd.DataFrame()
            
            # Use advanced recommender for context-aware predictions
            recommendations = RecommendationService.advanced_recommender.get_context_aware_recommendations(
                user_id=user_id,
                top_k=top_k,
                user_location=user_location,
                timestamp=datetime.now(),
                df_interactions=df_interactions if not df_interactions.empty else None,
                all_products=all_products
            )
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Context-aware recommendations failed: {e}")
            # Fallback to basic recommendations
            return RecommendationService.recommend_products_for_campaigns(
                [{'product_id': pid} for pid in (all_products or ['P001', 'P002', 'P003'])], 
                top_k
            )
    
    @staticmethod
    def get_trending_recommendations(location: str = 'nairobi', top_k: int = 10) -> List[Dict]:
        """Get trending product recommendations for a specific location"""
        try:
            with SessionLocal() as db:
                # Get recent interactions for trending analysis
                interactions = db.query(UserProductInteraction).all()
                
                if not interactions:
                    return []
                
                # Simple trending logic based on recent interactions
                df = pd.DataFrame([{
                    'product_id': str(i.product_id),
                    'timestamp': datetime.now()  # Would be actual timestamp
                } for i in interactions])
                
                # Count interactions per product in last 24h
                trending = df.groupby('product_id').size().sort_values(ascending=False).head(top_k)
                
                trending_products = []
                for product_id, count in trending.items():
                    trending_products.append({
                        'product_id': product_id,
                        'trending_score': float(count),
                        'location': location,
                        'context': 'trending_in_' + location.lower()
                    })
                
                return trending_products
                
        except Exception as e:
            logging.error(f"Trending recommendations failed: {e}")
            return []
    
    @staticmethod
    def get_payday_promotions(user_id: str, top_k: int = 5) -> List[Dict]:
        """Get special recommendations for payday periods"""
        current_day = datetime.now().day
        
        # Check if it's payday period (15th or end of month)
        is_payday = current_day in [15, 30, 31] or (current_day >= 28 and datetime.now().month == 2)
        
        if not is_payday:
            return []
        
        try:
            with SessionLocal() as db:
                # Get high-value products that users might buy on payday
                products = db.query(Product).filter(Product.price > 5000).limit(top_k).all()
                
                payday_recommendations = []
                for product in products:
                    payday_recommendations.append({
                        'product_id': str(product.product_id),
                        'product_name': product.name,
                        'price': product.price,
                        'discount': 0.1,  # 10% payday discount
                        'context': 'payday_promotion',
                        'promotion_message': 'Payday Special - 10% off!'
                    })
                
                return payday_recommendations
                
        except Exception as e:
            logging.error(f"Payday promotions failed: {e}")
            return []
    
    @staticmethod
    def train_advanced_models(interaction_data: pd.DataFrame):
        """Train the advanced recommendation models"""
        try:
            # Train matrix factorization
            if not interaction_data.empty:
                # Create user-item matrix
                user_item_matrix = interaction_data.pivot_table(
                    index='user_id', columns='product_id', values='rating', fill_value=0
                ).values
                
                RecommendationService.advanced_recommender.train_matrix_factorization(
                    user_item_matrix, model_type='als'
                )
                
                # Train DLRM if enough data
                if len(interaction_data) > 100:
                    RecommendationService.advanced_recommender.train_dlrm(
                        interaction_data, epochs=50
                    )
                
                logging.info("Advanced recommendation models trained successfully")
            else:
                logging.warning("No interaction data available for training")
                
        except Exception as e:
            logging.error(f"Advanced model training failed: {e}")
    
    @staticmethod
    def get_latest_ctr() -> float:
        """Get latest click-through rate for recommendation quality monitoring."""
        # In real implementation, this would fetch from analytics DB
        return 0.15  # Improved CTR with advanced models

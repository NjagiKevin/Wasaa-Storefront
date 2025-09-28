"""
Advanced Recommendation Engine with Matrix Factorization, DLRM, and Context-Aware Features
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from surprise import Dataset, Reader, SVD, NMF
from surprise.model_selection import train_test_split, GridSearchCV
import logging
import mlflow
import joblib
from datetime import datetime, timedelta
import holidays
from geopy.distance import geodesic
import warnings
warnings.filterwarnings("ignore")

class DeepLearningRecommenderModel(nn.Module):
    """
    Deep Learning Recommender Model (DLRM) implementation
    """
    def __init__(self, num_users: int, num_items: int, num_features: int, 
                 embedding_dim: int = 64, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Feature processing
        self.feature_mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Interaction layers
        interaction_input_dim = 2 * embedding_dim + hidden_dims[1]
        self.interaction_mlp = nn.Sequential(
            nn.Linear(interaction_input_dim, hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[2], 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, features: torch.Tensor):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Process features
        feature_emb = self.feature_mlp(features)
        
        # Concatenate all features
        interaction_input = torch.cat([user_emb, item_emb, feature_emb], dim=1)
        
        # Predict interaction probability
        output = self.interaction_mlp(interaction_input)
        return output.squeeze()

class ContextAwareRecommender:
    """
    Context-aware recommendation system with location, time, and trending features
    """
    
    def __init__(self):
        self.matrix_factorization_model = None
        self.dlrm_model = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        self.kenya_holidays = holidays.Kenya()
        
    def extract_temporal_features(self, timestamp: datetime) -> Dict[str, float]:
        """Extract time-based features"""
        features = {
            'hour_of_day': timestamp.hour / 24.0,
            'day_of_week': timestamp.weekday() / 6.0,
            'day_of_month': timestamp.day / 31.0,
            'month': timestamp.month / 12.0,
            'is_weekend': float(timestamp.weekday() >= 5),
            'is_holiday': float(timestamp.date() in self.kenya_holidays),
            'is_payday': float(timestamp.day in [15, 30, 31]),  # Common payday periods
            'quarter': float((timestamp.month - 1) // 3) / 3.0
        }
        return features
    
    def extract_location_features(self, user_location: Tuple[float, float], 
                                item_location: Tuple[float, float] = None) -> Dict[str, float]:
        """Extract location-based features"""
        nairobi_center = (-1.2921, 36.8219)  # Nairobi coordinates
        
        # Distance from Nairobi center
        distance_from_nairobi = geodesic(user_location, nairobi_center).kilometers
        
        features = {
            'distance_from_nairobi': min(distance_from_nairobi / 100.0, 1.0),  # Normalize
            'is_nairobi': float(distance_from_nairobi < 20),  # Within Nairobi area
            'latitude_normalized': (user_location[0] + 90) / 180.0,  # Normalize latitude
            'longitude_normalized': (user_location[1] + 180) / 360.0,  # Normalize longitude
        }
        
        # If item has location, add distance between user and item
        if item_location:
            user_item_distance = geodesic(user_location, item_location).kilometers
            features['user_item_distance'] = min(user_item_distance / 50.0, 1.0)
        
        return features
    
    def extract_trending_features(self, product_id: str, df_interactions: pd.DataFrame) -> Dict[str, float]:
        """Extract trending and popularity features"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_week = now - timedelta(days=7)
        
        # Recent interactions for this product
        recent_interactions = df_interactions[
            (df_interactions['product_id'] == product_id) & 
            (df_interactions['timestamp'] >= last_24h)
        ]
        
        weekly_interactions = df_interactions[
            (df_interactions['product_id'] == product_id) & 
            (df_interactions['timestamp'] >= last_week)
        ]
        
        features = {
            'interactions_24h': min(len(recent_interactions) / 100.0, 1.0),
            'interactions_7d': min(len(weekly_interactions) / 500.0, 1.0),
            'trending_score': min(len(recent_interactions) / max(1, len(weekly_interactions) / 7), 1.0),
            'global_popularity': min(len(df_interactions[df_interactions['product_id'] == product_id]) / 1000.0, 1.0)
        }
        
        return features
    
    def prepare_features(self, user_id: str, product_id: str, 
                        user_location: Tuple[float, float] = None,
                        product_location: Tuple[float, float] = None,
                        timestamp: datetime = None,
                        df_interactions: pd.DataFrame = None) -> np.ndarray:
        """Prepare comprehensive feature vector"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Extract all feature types
        temporal_features = self.extract_temporal_features(timestamp)
        
        location_features = {}
        if user_location:
            location_features = self.extract_location_features(user_location, product_location)
        
        trending_features = {}
        if df_interactions is not None:
            trending_features = self.extract_trending_features(product_id, df_interactions)
        
        # Combine all features
        all_features = {**temporal_features, **location_features, **trending_features}
        
        # Convert to array
        feature_vector = np.array(list(all_features.values()))
        
        return feature_vector
    
    def train_matrix_factorization(self, interaction_matrix: np.ndarray, model_type: str = 'als'):
        """Train matrix factorization model"""
        
        with mlflow.start_run(run_name=f"matrix_factorization_{model_type}"):
            if model_type == 'als':
                self.matrix_factorization_model = AlternatingLeastSquares(
                    factors=100, 
                    regularization=0.01, 
                    iterations=50,
                    random_state=42
                )
            elif model_type == 'bpr':
                self.matrix_factorization_model = BayesianPersonalizedRanking(
                    factors=100,
                    learning_rate=0.01,
                    regularization=0.01,
                    iterations=100,
                    random_state=42
                )
            
            # Convert to sparse matrix if needed
            from scipy.sparse import csr_matrix
            if not hasattr(interaction_matrix, 'nnz'):
                interaction_matrix = csr_matrix(interaction_matrix)
            
            self.matrix_factorization_model.fit(interaction_matrix)
            
            # Log model
            mlflow.sklearn.log_model(self.matrix_factorization_model, "matrix_factorization_model")
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("factors", 100)
            
        logging.info(f"Matrix factorization model ({model_type}) trained successfully")
    
    def train_dlrm(self, df_interactions: pd.DataFrame, epochs: int = 50):
        """Train Deep Learning Recommender Model"""
        
        # Prepare data
        users = df_interactions['user_id'].unique()
        items = df_interactions['product_id'].unique()
        
        self.user_encoder.fit(users)
        self.item_encoder.fit(items)
        
        # Encode users and items
        user_ids = self.user_encoder.transform(df_interactions['user_id'])
        item_ids = self.item_encoder.transform(df_interactions['product_id'])
        
        # Prepare features for each interaction
        features_list = []
        for idx, row in df_interactions.iterrows():
            feature_vector = self.prepare_features(
                row['user_id'], 
                row['product_id'],
                user_location=row.get('user_location'),
                product_location=row.get('product_location'),
                timestamp=row.get('timestamp'),
                df_interactions=df_interactions
            )
            features_list.append(feature_vector)
        
        features = np.array(features_list)
        if len(features) > 0:
            features = self.feature_scaler.fit_transform(features)
        
        # Create DLRM model
        num_users = len(users)
        num_items = len(items)
        num_features = features.shape[1] if len(features) > 0 else 8
        
        self.dlrm_model = DeepLearningRecommenderModel(
            num_users=num_users,
            num_items=num_items,
            num_features=num_features
        )
        
        # Convert to tensors
        user_tensor = torch.LongTensor(user_ids)
        item_tensor = torch.LongTensor(item_ids)
        feature_tensor = torch.FloatTensor(features if len(features) > 0 else np.zeros((len(user_ids), num_features)))
        target_tensor = torch.FloatTensor(df_interactions.get('rating', [1.0] * len(df_interactions)))
        
        # Train model
        optimizer = torch.optim.Adam(self.dlrm_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        with mlflow.start_run(run_name="dlrm_training"):
            for epoch in range(epochs):
                optimizer.zero_grad()
                predictions = self.dlrm_model(user_tensor, item_tensor, feature_tensor)
                loss = criterion(predictions, target_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    mlflow.log_metric("loss", loss.item(), step=epoch)
                    logging.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Save model
            torch.save(self.dlrm_model.state_dict(), "dlrm_model.pth")
            mlflow.log_artifact("dlrm_model.pth")
            
        logging.info("DLRM model trained successfully")
    
    def get_context_aware_recommendations(self, user_id: str, top_k: int = 10,
                                        user_location: Tuple[float, float] = None,
                                        timestamp: datetime = None,
                                        df_interactions: pd.DataFrame = None,
                                        all_products: List[str] = None) -> List[Dict[str, float]]:
        """Generate context-aware recommendations"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        recommendations = []
        
        if all_products is None:
            all_products = ['P001', 'P002', 'P003', 'P004', 'P005']  # Default products
        
        for product_id in all_products:
            score = 0.0
            
            # Matrix Factorization Score
            if self.matrix_factorization_model:
                try:
                    if hasattr(self.user_encoder, 'classes_') and user_id in self.user_encoder.classes_:
                        user_idx = self.user_encoder.transform([user_id])[0]
                        if hasattr(self.item_encoder, 'classes_') and product_id in self.item_encoder.classes_:
                            item_idx = self.item_encoder.transform([product_id])[0]
                            mf_score = self.matrix_factorization_model.recommend(
                                user_idx, csr_matrix((1, len(all_products))), N=1
                            )[0][1] if hasattr(self.matrix_factorization_model, 'recommend') else 0.5
                            score += 0.4 * mf_score
                except:
                    score += 0.4 * 0.5  # Default score
            
            # DLRM Score
            if self.dlrm_model:
                try:
                    features = self.prepare_features(
                        user_id, product_id, user_location, 
                        timestamp=timestamp, df_interactions=df_interactions
                    )
                    
                    if hasattr(self.user_encoder, 'classes_') and user_id in self.user_encoder.classes_:
                        user_idx = self.user_encoder.transform([user_id])[0]
                    else:
                        user_idx = 0
                    
                    if hasattr(self.item_encoder, 'classes_') and product_id in self.item_encoder.classes_:
                        item_idx = self.item_encoder.transform([product_id])[0]
                    else:
                        item_idx = 0
                    
                    features_scaled = self.feature_scaler.transform([features]) if hasattr(self.feature_scaler, 'scale_') else [features]
                    
                    with torch.no_grad():
                        user_tensor = torch.LongTensor([user_idx])
                        item_tensor = torch.LongTensor([item_idx])
                        feature_tensor = torch.FloatTensor(features_scaled)
                        
                        dlrm_score = self.dlrm_model(user_tensor, item_tensor, feature_tensor).item()
                        score += 0.6 * dlrm_score
                except Exception as e:
                    logging.warning(f"DLRM prediction failed for {user_id}, {product_id}: {e}")
                    score += 0.6 * 0.5  # Default score
            
            recommendations.append({
                'product_id': product_id,
                'score': min(max(score, 0.0), 1.0),  # Clamp between 0 and 1
                'context': {
                    'timestamp': timestamp.isoformat(),
                    'user_location': user_location,
                    'is_trending': self.is_trending_now(product_id, timestamp) if df_interactions is not None else False,
                    'is_contextually_relevant': self.is_contextually_relevant(user_location, timestamp)
                }
            })
        
        # Sort by score and return top_k
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]
    
    def is_trending_now(self, product_id: str, timestamp: datetime, 
                       df_interactions: pd.DataFrame = None) -> bool:
        """Check if product is currently trending"""
        if df_interactions is None:
            return False
        
        last_24h = timestamp - timedelta(hours=24)
        recent_interactions = df_interactions[
            (df_interactions['product_id'] == product_id) & 
            (df_interactions['timestamp'] >= last_24h)
        ]
        
        # Trending if more than 5 interactions in last 24h
        return len(recent_interactions) > 5
    
    def is_contextually_relevant(self, user_location: Tuple[float, float] = None, 
                               timestamp: datetime = None) -> bool:
        """Check if recommendation is contextually relevant"""
        if timestamp is None:
            return True
        
        # More relevant during business hours and weekdays
        is_business_hours = 9 <= timestamp.hour <= 17
        is_weekday = timestamp.weekday() < 5
        
        return is_business_hours and is_weekday
    
    def save_model(self, model_path: str):
        """Save the trained models"""
        model_data = {
            'matrix_factorization_model': self.matrix_factorization_model,
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'feature_scaler': self.feature_scaler
        }
        
        joblib.dump(model_data, f"{model_path}/context_aware_recommender.pkl")
        
        if self.dlrm_model:
            torch.save(self.dlrm_model.state_dict(), f"{model_path}/dlrm_model.pth")
        
        logging.info(f"Models saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load the trained models"""
        try:
            model_data = joblib.load(f"{model_path}/context_aware_recommender.pkl")
            self.matrix_factorization_model = model_data['matrix_factorization_model']
            self.user_encoder = model_data['user_encoder']
            self.item_encoder = model_data['item_encoder']
            self.feature_scaler = model_data['feature_scaler']
            
            # Load DLRM model if exists
            dlrm_path = f"{model_path}/dlrm_model.pth"
            try:
                if self.dlrm_model:
                    self.dlrm_model.load_state_dict(torch.load(dlrm_path, map_location='cpu'))
            except:
                logging.warning("Could not load DLRM model")
            
            logging.info(f"Models loaded from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load models: {e}")

# Factory function for easy usage
def create_advanced_recommender() -> ContextAwareRecommender:
    """Create and return an advanced recommender instance"""
    return ContextAwareRecommender()
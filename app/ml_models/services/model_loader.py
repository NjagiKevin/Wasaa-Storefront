"""
Production Model Loading Service for Storefront ML

Enhanced model loading with MLflow Model Registry integration,
caching, and fallback mechanisms based on AdsManager best practices.
"""

import logging
import os
from typing import Optional, Dict, Any, Union
from functools import lru_cache
import joblib
import pickle

from ..tracking.mlflow_utils import (
    configure_mlflow,
    load_model_from_registry,
    get_model_version_by_stage,
    ModelStage
)

logger = logging.getLogger(__name__)

class StorefrontModelLoader:
    """Production-ready model loader using MLflow Model Registry."""
    
    # Registered model names for the storefront
    MODEL_REGISTRY = {
        "recommender": "storefront-recommender",
        "forecasting": "storefront-forecasting", 
        "fraud": "storefront-fraud-detection",
        "clv": "storefront-customer-lifetime-value",
        "churn": "storefront-churn-prediction",
        "bundling": "storefront-product-bundling"
    }
    
    def __init__(self):
        """Initialize model loader."""
        configure_mlflow()
        self._model_cache = {}
        self._fallback_enabled = os.getenv("MODEL_FALLBACK_ENABLED", "true").lower() == "true"
        self._cache_ttl = int(os.getenv("MODEL_CACHE_TTL", "3600"))  # 1 hour default
        
    @lru_cache(maxsize=20)
    def load_model(
        self, 
        model_name: str, 
        stage: ModelStage = ModelStage.PRODUCTION,
        version: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Load a model from MLflow Model Registry.
        
        Args:
            model_name: Model name (e.g., 'recommender', 'forecasting')
            stage: Stage to load from (ignored if version specified)
            version: Specific version to load
            use_cache: Whether to use cached models
            
        Returns:
            Loaded model or None if failed
        """
        # Get registered model name
        registered_name = self.MODEL_REGISTRY.get(model_name, model_name)
        cache_key = f"{registered_name}:{version or stage.value}"
        
        # Check cache first
        if use_cache and cache_key in self._model_cache:
            logger.info(f"📦 Loading cached model: {cache_key}")
            return self._model_cache[cache_key]
        
        try:
            # Load from registry
            model = load_model_from_registry(
                model_name=registered_name,
                stage=stage,
                version=version
            )
            
            if model:
                # Cache the model
                if use_cache:
                    self._model_cache[cache_key] = model
                
                logger.info(f"✅ Successfully loaded model: {cache_key}")
                return model
            else:
                logger.warning(f"⚠️  Failed to load model from registry: {cache_key}")
                
        except Exception as e:
            logger.error(f"❌ Error loading model from registry: {e}")
        
        # Fallback to local file if enabled
        if self._fallback_enabled:
            return self._load_fallback_model(model_name)
        
        return None
    
    def _load_fallback_model(self, model_name: str):
        """Load model from local fallback file."""
        try:
            fallback_paths = {
                "recommender": "models/storefront_recommender.pkl",
                "forecasting": "models/storefront_forecasting.pkl",
                "fraud": "models/storefront_fraud_detection.pkl",
                "clv": "models/storefront_clv_model.pkl",
                "churn": "models/storefront_churn_model.pkl", 
                "bundling": "models/storefront_bundling_model.pkl"
            }
            
            fallback_path = fallback_paths.get(model_name)
            if not fallback_path or not os.path.exists(fallback_path):
                logger.warning(f"⚠️  No fallback model found for {model_name}")
                return None
            
            # Load using appropriate method based on extension
            if fallback_path.endswith('.pkl'):
                with open(fallback_path, 'rb') as f:
                    model = pickle.load(f)
            elif fallback_path.endswith('.joblib'):
                model = joblib.load(fallback_path)
            else:
                logger.error(f"❌ Unsupported fallback model format: {fallback_path}")
                return None
            
            logger.info(f"📁 Loaded fallback model: {fallback_path}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load fallback model: {e}")
            return None
    
    def get_model_info(
        self, 
        model_name: str, 
        stage: ModelStage = ModelStage.PRODUCTION
    ) -> Optional[Dict[str, Any]]:
        """Get information about a model in the registry."""
        try:
            registered_name = self.MODEL_REGISTRY.get(model_name, model_name)
            return get_model_version_by_stage(registered_name, stage)
        except Exception as e:
            logger.error(f"❌ Failed to get model info for {model_name}: {e}")
            return None
    
    def clear_cache(self):
        """Clear the model cache."""
        self._model_cache.clear()
        # Clear LRU cache
        self.load_model.cache_clear()
        logger.info("🧹 Model cache cleared")
    
    def list_cached_models(self) -> Dict[str, Any]:
        """List currently cached models."""
        return {
            "cached_models": list(self._model_cache.keys()),
            "cache_size": len(self._model_cache),
            "cache_info": self.load_model.cache_info()._asdict()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on model loading system."""
        health_status = {
            "status": "healthy",
            "models_available": {},
            "registry_connection": True,
            "fallback_enabled": self._fallback_enabled,
            "cache_size": len(self._model_cache)
        }
        
        # Check each model's availability
        for model_name in self.MODEL_REGISTRY.keys():
            try:
                model_info = self.get_model_info(model_name, ModelStage.PRODUCTION)
                health_status["models_available"][model_name] = {
                    "registry": model_info is not None,
                    "fallback": self._check_fallback_availability(model_name),
                    "version": model_info.get("version") if model_info else None
                }
            except Exception as e:
                health_status["models_available"][model_name] = {
                    "registry": False,
                    "fallback": self._check_fallback_availability(model_name),
                    "error": str(e)
                }
                health_status["status"] = "degraded"
        
        return health_status
    
    def _check_fallback_availability(self, model_name: str) -> bool:
        """Check if fallback model is available."""
        fallback_paths = {
            "recommender": "models/storefront_recommender.pkl",
            "forecasting": "models/storefront_forecasting.pkl",
            "fraud": "models/storefront_fraud_detection.pkl",
            "clv": "models/storefront_clv_model.pkl",
            "churn": "models/storefront_churn_model.pkl",
            "bundling": "models/storefront_bundling_model.pkl"
        }
        
        fallback_path = fallback_paths.get(model_name)
        return fallback_path is not None and os.path.exists(fallback_path)

# Global model loader instance
model_loader = StorefrontModelLoader()

# Convenience functions for easy access
def get_production_model(model_name: str, version: Optional[str] = None):
    """
    Get a production model.
    
    Args:
        model_name: Model name (e.g., 'recommender', 'forecasting')
        version: Specific version (optional, uses Production stage if not specified)
        
    Returns:
        Loaded model or None
    """
    return model_loader.load_model(
        model_name=model_name,
        stage=ModelStage.PRODUCTION,
        version=version
    )

def get_staging_model(model_name: str, version: Optional[str] = None):
    """
    Get a staging model for testing.
    
    Args:
        model_name: Model name
        version: Specific version (optional, uses Staging stage if not specified)
        
    Returns:
        Loaded model or None
    """
    return model_loader.load_model(
        model_name=model_name,
        stage=ModelStage.STAGING,
        version=version
    )

def get_recommender_model(version: Optional[str] = None):
    """Get the production recommender model."""
    return get_production_model("recommender", version)

def get_forecasting_model(version: Optional[str] = None):
    """Get the production forecasting model.""" 
    return get_production_model("forecasting", version)

def get_fraud_detection_model(version: Optional[str] = None):
    """Get the production fraud detection model."""
    return get_production_model("fraud", version)

def get_clv_model(version: Optional[str] = None):
    """Get the production CLV model."""
    return get_production_model("clv", version)

def get_churn_model(version: Optional[str] = None):
    """Get the production churn prediction model."""
    return get_production_model("churn", version)

def get_bundling_model(version: Optional[str] = None):
    """Get the production product bundling model."""
    return get_production_model("bundling", version)

def reload_models():
    """Reload all models (clear cache and force reload)."""
    model_loader.clear_cache()
    logger.info("🔄 Models reloaded")

def get_model_health():
    """Get health status of all models."""
    return model_loader.health_check()
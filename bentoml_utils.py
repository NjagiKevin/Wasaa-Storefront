"""
BentoML Model Management Utilities
Provides utilities for saving and loading ML models to/from BentoML model store
"""

import os
import bentoml
import joblib
import logging
from typing import Any, Dict, Optional
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages ML model persistence with BentoML"""
    
    def __init__(self):
        self.model_store_path = os.getenv("BENTOML_HOME", "/bentoml")
        
    def save_sklearn_model(self, model: Any, model_name: str, 
                          metadata: Optional[Dict] = None) -> bool:
        """Save scikit-learn compatible model to BentoML store"""
        try:
            # Add default metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "framework": "sklearn",
                "saved_at": datetime.now().isoformat(),
                "model_type": type(model).__name__
            })
            
            # Save model
            bentoml.sklearn.save_model(
                name=model_name,
                model=model,
                metadata=metadata,
                signatures={
                    "predict": {"batchable": True},
                    "predict_proba": {"batchable": True}
                }
            )
            
            logger.info(f"Successfully saved sklearn model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save sklearn model {model_name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_sklearn_model(self, model_name: str, version: Optional[str] = None) -> Optional[Any]:
        """Load scikit-learn model from BentoML store"""
        try:
            model_tag = f"{model_name}:{version}" if version else f"{model_name}:latest"
            model = bentoml.sklearn.load_model(model_tag)
            
            logger.info(f"Successfully loaded sklearn model: {model_tag}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load sklearn model {model_name}: {e}")
            return None
    
    def save_pytorch_model(self, model: Any, model_name: str, 
                          metadata: Optional[Dict] = None) -> bool:
        """Save PyTorch model to BentoML store"""
        try:
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "framework": "pytorch",
                "saved_at": datetime.now().isoformat(),
                "model_type": type(model).__name__
            })
            
            # Save PyTorch model
            bentoml.pytorch.save_model(
                name=model_name,
                model=model,
                metadata=metadata
            )
            
            logger.info(f"Successfully saved PyTorch model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save PyTorch model {model_name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_pytorch_model(self, model_name: str, version: Optional[str] = None) -> Optional[Any]:
        """Load PyTorch model from BentoML store"""
        try:
            model_tag = f"{model_name}:{version}" if version else f"{model_name}:latest"
            model = bentoml.pytorch.load_model(model_tag)
            
            logger.info(f"Successfully loaded PyTorch model: {model_tag}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model {model_name}: {e}")
            return None
    
    def save_custom_model(self, model_data: Any, model_name: str, 
                         framework: str = "custom", 
                         metadata: Optional[Dict] = None) -> bool:
        """Save custom model using joblib serialization"""
        try:
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "framework": framework,
                "saved_at": datetime.now().isoformat(),
                "serialization": "joblib"
            })
            
            # Save using BentoML's general model saving
            bentoml.models.create(
                name=model_name,
                module=__name__,
                api_version="v1",
                metadata=metadata,
                context={"model_data": model_data}
            )
            
            logger.info(f"Successfully saved custom model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save custom model {model_name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_custom_model(self, model_name: str, version: Optional[str] = None) -> Optional[Any]:
        """Load custom model"""
        try:
            model_tag = f"{model_name}:{version}" if version else f"{model_name}:latest"
            model_ref = bentoml.models.get(model_tag)
            
            # Load the actual model data
            model_data = model_ref.context.get("model_data")
            
            logger.info(f"Successfully loaded custom model: {model_tag}")
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to load custom model {model_name}: {e}")
            return None
    
    def list_models(self, model_name: Optional[str] = None) -> list:
        """List available models in BentoML store"""
        try:
            if model_name:
                models = bentoml.models.list(model_name)
            else:
                models = bentoml.models.list()
            
            model_info = []
            for model in models:
                model_info.append({
                    "name": model.tag.name,
                    "version": model.tag.version,
                    "created_at": model.info.creation_time,
                    "size": model.info.size_bytes,
                    "framework": model.info.metadata.get("framework", "unknown")
                })
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def delete_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Delete model from BentoML store"""
        try:
            model_tag = f"{model_name}:{version}" if version else f"{model_name}:latest"
            bentoml.models.delete(model_tag)
            
            logger.info(f"Successfully deleted model: {model_tag}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False

# Global model manager instance
model_manager = ModelManager()

def save_recommendation_models(models: Dict[str, Any]) -> bool:
    """Save recommendation models to BentoML store"""
    success = True
    
    for model_name, model_obj in models.items():
        metadata = {
            "model_type": "recommendation",
            "algorithm": model_name,
            "description": f"Recommendation model using {model_name}"
        }
        
        if hasattr(model_obj, 'fit'):  # Sklearn-like model
            success &= model_manager.save_sklearn_model(
                model_obj, f"recommendation_{model_name}", metadata
            )
        else:  # Custom model
            success &= model_manager.save_custom_model(
                model_obj, f"recommendation_{model_name}", 
                framework="recommendation", metadata=metadata
            )
    
    return success

def save_forecasting_models(models: Dict[str, Any]) -> bool:
    """Save forecasting models to BentoML store"""
    success = True
    
    for model_name, model_obj in models.items():
        metadata = {
            "model_type": "forecasting", 
            "algorithm": model_name,
            "description": f"Demand forecasting model using {model_name}"
        }
        
        if "pytorch" in str(type(model_obj)).lower():
            success &= model_manager.save_pytorch_model(
                model_obj, f"forecasting_{model_name}", metadata
            )
        elif hasattr(model_obj, 'fit'):
            success &= model_manager.save_sklearn_model(
                model_obj, f"forecasting_{model_name}", metadata
            )
        else:
            success &= model_manager.save_custom_model(
                model_obj, f"forecasting_{model_name}",
                framework="forecasting", metadata=metadata
            )
    
    return success

def save_fraud_models(models: Dict[str, Any]) -> bool:
    """Save fraud detection models to BentoML store"""
    success = True
    
    for model_name, model_obj in models.items():
        metadata = {
            "model_type": "fraud_detection",
            "algorithm": model_name, 
            "description": f"Fraud detection model using {model_name}"
        }
        
        if hasattr(model_obj, 'predict_proba'):  # Sklearn classifier
            success &= model_manager.save_sklearn_model(
                model_obj, f"fraud_{model_name}", metadata
            )
        else:
            success &= model_manager.save_custom_model(
                model_obj, f"fraud_{model_name}",
                framework="fraud_detection", metadata=metadata
            )
    
    return success

def load_models_for_service() -> Dict[str, Any]:
    """Load all available models for the service"""
    models = {
        "recommendation": {},
        "forecasting": {},
        "fraud_detection": {}
    }
    
    try:
        all_models = model_manager.list_models()
        
        for model_info in all_models:
            model_name = model_info["name"]
            
            if model_name.startswith("recommendation_"):
                algorithm = model_name.replace("recommendation_", "")
                model = model_manager.load_sklearn_model(model_name)
                if model:
                    models["recommendation"][algorithm] = model
                    
            elif model_name.startswith("forecasting_"):
                algorithm = model_name.replace("forecasting_", "")
                # Try different loading methods
                model = (model_manager.load_pytorch_model(model_name) or 
                        model_manager.load_sklearn_model(model_name) or
                        model_manager.load_custom_model(model_name))
                if model:
                    models["forecasting"][algorithm] = model
                    
            elif model_name.startswith("fraud_"):
                algorithm = model_name.replace("fraud_", "")
                model = (model_manager.load_sklearn_model(model_name) or
                        model_manager.load_custom_model(model_name))
                if model:
                    models["fraud_detection"][algorithm] = model
        
        logger.info(f"Loaded models: {len(models['recommendation'])} recommendation, "
                   f"{len(models['forecasting'])} forecasting, "
                   f"{len(models['fraud_detection'])} fraud detection")
        
    except Exception as e:
        logger.error(f"Failed to load models for service: {e}")
    
    return models

if __name__ == "__main__":
    # Test the model manager
    logging.basicConfig(level=logging.INFO)
    
    # List available models
    models = model_manager.list_models()
    print(f"Available models: {len(models)}")
    for model in models:
        print(f"- {model['name']}:{model['version']} ({model['framework']})")
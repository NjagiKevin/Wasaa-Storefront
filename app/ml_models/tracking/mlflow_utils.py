"""
Production MLflow Model Registry Utilities

Based on best practices from AdsManager ML system.
Provides enterprise-grade model versioning, staging, and deployment.
"""

import logging
import os
from typing import Optional, Dict, Any, List
from enum import Enum
from functools import wraps

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

class ModelStage(Enum):
    """Model stages in MLflow Model Registry."""
    NONE = "None"
    STAGING = "Staging" 
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

def configure_mlflow():
    """Configure MLflow with environment variables."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", tracking_uri)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "storefront-ml")
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    
    # Create experiment if it doesn't exist
    try:
        mlflow.create_experiment(experiment_name)
    except MlflowException:
        pass  # Experiment already exists
    
    mlflow.set_experiment(experiment_name)
    logger.info(f"🔧 MLflow configured - Tracking: {tracking_uri}, Registry: {registry_uri}")

def mlflow_run(func):
    """Decorator to automatically start/end MLflow runs."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not mlflow.active_run():
            with mlflow.start_run():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper

def register_model(
    model_name: str,
    run_id: str,
    model_path: str = "model",
    description: str = None,
    tags: Dict[str, str] = None
) -> Optional[str]:
    """
    Register a model in MLflow Model Registry.
    
    Args:
        model_name: Name for the registered model
        run_id: MLflow run ID containing the model
        model_path: Path within the run artifacts
        description: Model description
        tags: Model tags
        
    Returns:
        Model version string or None if failed
    """
    try:
        configure_mlflow()
        client = MlflowClient()
        
        # Create registered model if it doesn't exist
        try:
            client.create_registered_model(
                name=model_name,
                description=description or f"Storefront ML model: {model_name}"
            )
            logger.info(f"📝 Created registered model: {model_name}")
        except MlflowException:
            logger.info(f"📋 Model {model_name} already registered")
        
        # Register model version
        model_uri = f"runs:/{run_id}/{model_path}"
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
            description=description
        )
        
        # Add tags if provided
        if tags:
            for key, value in tags.items():
                client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=key,
                    value=value
                )
        
        # Automatically transition to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=ModelStage.STAGING.value,
            archive_existing_versions=False
        )
        
        logger.info(f"✅ Registered model {model_name} v{model_version.version} -> Staging")
        return model_version.version
        
    except Exception as e:
        logger.error(f"❌ Failed to register model {model_name}: {e}")
        return None

def promote_model_to_production(
    model_name: str,
    version: str,
    approval_comment: str = "Automated promotion"
) -> bool:
    """
    Promote a model version to production.
    
    Args:
        model_name: Registered model name
        version: Version to promote
        approval_comment: Comment for the promotion
        
    Returns:
        True if successful, False otherwise
    """
    try:
        configure_mlflow()
        client = MlflowClient()
        
        # Archive existing production versions
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=ModelStage.PRODUCTION.value,
            archive_existing_versions=True,
            description=approval_comment
        )
        
        logger.info(f"🚀 Promoted {model_name} v{version} to Production")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to promote model {model_name} v{version}: {e}")
        return False

def load_model_from_registry(
    model_name: str,
    stage: ModelStage = ModelStage.PRODUCTION,
    version: Optional[str] = None
):
    """
    Load a model from MLflow Model Registry.
    
    Args:
        model_name: Registered model name
        stage: Stage to load from (ignored if version specified)
        version: Specific version to load
        
    Returns:
        Loaded model or None if failed
    """
    try:
        configure_mlflow()
        
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/{stage.value}"
        
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"📦 Loaded model: {model_uri}")
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load model {model_uri}: {e}")
        return None

def get_model_version_by_stage(
    model_name: str,
    stage: ModelStage
) -> Optional[Dict[str, Any]]:
    """
    Get model version information by stage.
    
    Args:
        model_name: Registered model name
        stage: Stage to query
        
    Returns:
        Model version info dict or None
    """
    try:
        configure_mlflow()
        client = MlflowClient()
        
        versions = client.get_latest_versions(
            name=model_name,
            stages=[stage.value]
        )
        
        if not versions:
            logger.warning(f"⚠️ No model found in {stage.value} stage for {model_name}")
            return None
        
        version = versions[0]
        return {
            "name": version.name,
            "version": version.version,
            "stage": version.current_stage,
            "description": version.description,
            "creation_timestamp": version.creation_timestamp,
            "last_updated_timestamp": version.last_updated_timestamp,
            "source": version.source,
            "run_id": version.run_id
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get model version for {model_name}: {e}")
        return None

def list_registered_models() -> List[Dict[str, Any]]:
    """
    List all registered models.
    
    Returns:
        List of model info dictionaries
    """
    try:
        configure_mlflow()
        client = MlflowClient()
        
        models = []
        for model in client.list_registered_models():
            model_info = {
                "name": model.name,
                "description": model.description,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "latest_versions": []
            }
            
            # Get latest versions for each stage
            for stage in ModelStage:
                try:
                    versions = client.get_latest_versions(
                        name=model.name,
                        stages=[stage.value]
                    )
                    if versions:
                        version_info = {
                            "version": versions[0].version,
                            "stage": versions[0].current_stage,
                            "creation_timestamp": versions[0].creation_timestamp
                        }
                        model_info["latest_versions"].append(version_info)
                except:
                    continue
            
            models.append(model_info)
        
        logger.info(f"📋 Found {len(models)} registered models")
        return models
        
    except Exception as e:
        logger.error(f"❌ Failed to list models: {e}")
        return []

def get_model_performance_metrics(
    model_name: str,
    stage: ModelStage = ModelStage.PRODUCTION
) -> Dict[str, Any]:
    """
    Get performance metrics for a model version.
    
    Args:
        model_name: Registered model name
        stage: Stage to get metrics for
        
    Returns:
        Dictionary of metrics
    """
    try:
        configure_mlflow()
        client = MlflowClient()
        
        # Get model version
        version_info = get_model_version_by_stage(model_name, stage)
        if not version_info:
            return {}
        
        # Get run metrics
        run = client.get_run(version_info["run_id"])
        metrics = run.data.metrics
        
        # Add model metadata
        metrics.update({
            "model_name": model_name,
            "version": version_info["version"],
            "stage": version_info["stage"],
            "creation_timestamp": version_info["creation_timestamp"]
        })
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Failed to get metrics for {model_name}: {e}")
        return {}

def compare_model_performance(
    model_name: str,
    metric_name: str = "accuracy"
) -> Dict[str, Any]:
    """
    Compare model performance across stages.
    
    Args:
        model_name: Registered model name
        metric_name: Metric to compare
        
    Returns:
        Comparison results
    """
    try:
        staging_metrics = get_model_performance_metrics(model_name, ModelStage.STAGING)
        production_metrics = get_model_performance_metrics(model_name, ModelStage.PRODUCTION)
        
        comparison = {
            "model_name": model_name,
            "metric_name": metric_name,
            "staging": {
                "value": staging_metrics.get(metric_name),
                "version": staging_metrics.get("version")
            },
            "production": {
                "value": production_metrics.get(metric_name),
                "version": production_metrics.get("version")
            }
        }
        
        # Calculate improvement
        if (staging_metrics.get(metric_name) is not None and 
            production_metrics.get(metric_name) is not None):
            staging_val = staging_metrics[metric_name]
            production_val = production_metrics[metric_name]
            improvement = ((staging_val - production_val) / production_val) * 100
            comparison["improvement_percent"] = improvement
            comparison["recommendation"] = "PROMOTE" if improvement > 0 else "KEEP_CURRENT"
        
        return comparison
        
    except Exception as e:
        logger.error(f"❌ Failed to compare models: {e}")
        return {}

# Initialize MLflow when module is imported
configure_mlflow()
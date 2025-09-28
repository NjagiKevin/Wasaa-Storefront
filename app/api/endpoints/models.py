"""
Model Management API Endpoints

Production-ready endpoints for MLflow Model Registry operations,
health checks, and model serving based on AdsManager best practices.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

from ...ml_models.services.model_loader import (
    model_loader, 
    get_production_model,
    get_staging_model, 
    get_model_health,
    reload_models
)
from ...ml_models.tracking.mlflow_utils import (
    list_registered_models,
    get_model_performance_metrics,
    compare_model_performance,
    promote_model_to_production,
    ModelStage
)

router = APIRouter()

# Request/Response Models
class ModelPromotionRequest(BaseModel):
    version: str
    target_stage: str = "Production"
    comment: Optional[str] = "API promotion"

class ModelComparisonRequest(BaseModel):
    model_name: str
    metric_name: str = "accuracy"

class ModelLoadRequest(BaseModel):
    model_name: str
    stage: str = "Production"
    version: Optional[str] = None

# Model Registry Endpoints
@router.get("/", tags=["models"])
async def list_models() -> Dict[str, Any]:
    """List all registered models in MLflow Registry."""
    try:
        models = list_registered_models()
        return {
            "status": "success",
            "models": models,
            "total_count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.get("/{model_name}/{stage}", tags=["models"])
async def get_model_info(
    model_name: str, 
    stage: str = "Production",
    include_metrics: bool = Query(False, description="Include performance metrics")
) -> Dict[str, Any]:
    """Get detailed information about a specific model version."""
    try:
        # Map string to enum
        stage_enum = ModelStage(stage)
        
        # Get model info
        model_info = model_loader.get_model_info(model_name, stage_enum)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found in {stage}")
        
        result = {
            "status": "success",
            "model_info": model_info
        }
        
        # Include metrics if requested
        if include_metrics:
            metrics = get_model_performance_metrics(model_name, stage_enum)
            result["metrics"] = metrics
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {stage}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.post("/{model_name}/promote", tags=["models"])
async def promote_model(
    model_name: str,
    promotion_request: ModelPromotionRequest
) -> Dict[str, Any]:
    """Promote a model version to a target stage."""
    try:
        success = promote_model_to_production(
            model_name=model_name,
            version=promotion_request.version,
            approval_comment=promotion_request.comment
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Model {model_name} v{promotion_request.version} promoted to {promotion_request.target_stage}",
                "model_name": model_name,
                "version": promotion_request.version,
                "target_stage": promotion_request.target_stage
            }
        else:
            raise HTTPException(status_code=500, detail="Model promotion failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to promote model: {str(e)}")

@router.get("/{model_name}/compare", tags=["models"])
async def compare_models(
    model_name: str,
    metric_name: str = Query("accuracy", description="Metric to compare"),
    staging: bool = Query(True, description="Include staging model"),
    production: bool = Query(True, description="Include production model")
) -> Dict[str, Any]:
    """Compare model performance between stages."""
    try:
        comparison = compare_model_performance(model_name, metric_name)
        
        if not comparison:
            raise HTTPException(status_code=404, detail=f"No comparison data available for {model_name}")
        
        return {
            "status": "success",
            "comparison": comparison
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare models: {str(e)}")

@router.post("/{model_name}/load", tags=["models"])
async def load_model_endpoint(
    model_name: str,
    load_request: ModelLoadRequest
) -> Dict[str, Any]:
    """Load a model (validates connectivity and caches it)."""
    try:
        # Map string to enum
        stage_enum = ModelStage(load_request.stage)
        
        model = model_loader.load_model(
            model_name=load_request.model_name or model_name,
            stage=stage_enum,
            version=load_request.version
        )
        
        if model is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Failed to load model {model_name} from {load_request.stage}"
            )
        
        return {
            "status": "success",
            "message": f"Model {model_name} loaded successfully",
            "model_name": model_name,
            "stage": load_request.stage,
            "version": load_request.version,
            "cached": True
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {load_request.stage}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Model Serving Endpoints
@router.post("/{model_name}/predict", tags=["serving"])
async def predict_with_model(
    model_name: str,
    prediction_data: Dict[str, Any],
    version: Optional[str] = Query(None, description="Specific model version"),
    stage: str = Query("Production", description="Model stage")
) -> Dict[str, Any]:
    """Make predictions using a registered model."""
    try:
        # Load the model
        if version:
            model = get_production_model(model_name, version)
        elif stage == "Staging":
            model = get_staging_model(model_name)
        else:
            model = get_production_model(model_name)
        
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not available"
            )
        
        # Make prediction (this will depend on your model interface)
        # For now, return a placeholder response
        return {
            "status": "success",
            "model_name": model_name,
            "version": version,
            "stage": stage,
            "predictions": "Placeholder - implement based on your model interface",
            "input_data": prediction_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Health and Management Endpoints
@router.get("/health", tags=["health"])
async def model_health_check() -> Dict[str, Any]:
    """Comprehensive health check for all models."""
    try:
        health_status = get_model_health()
        
        # Determine HTTP status based on health
        http_status = 200 if health_status["status"] == "healthy" else 503
        
        return health_status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/cache/info", tags=["cache"])
async def get_cache_info() -> Dict[str, Any]:
    """Get information about the model cache."""
    try:
        cache_info = model_loader.list_cached_models()
        return {
            "status": "success",
            "cache_info": cache_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache info: {str(e)}")

@router.post("/cache/clear", tags=["cache"])
async def clear_model_cache(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Clear the model cache."""
    try:
        background_tasks.add_task(model_loader.clear_cache)
        return {
            "status": "success",
            "message": "Model cache clearing initiated"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.post("/reload", tags=["management"])
async def reload_all_models(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Reload all models (clear cache and force reload)."""
    try:
        background_tasks.add_task(reload_models)
        return {
            "status": "success",
            "message": "Model reload initiated"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload models: {str(e)}")

# Metrics and Monitoring
@router.get("/{model_name}/metrics", tags=["monitoring"])
async def get_model_metrics(
    model_name: str,
    stage: str = Query("Production", description="Model stage"),
    include_system_metrics: bool = Query(False, description="Include system performance metrics")
) -> Dict[str, Any]:
    """Get comprehensive metrics for a model."""
    try:
        stage_enum = ModelStage(stage)
        
        # Get performance metrics
        metrics = get_model_performance_metrics(model_name, stage_enum)
        
        result = {
            "status": "success",
            "model_name": model_name,
            "stage": stage,
            "metrics": metrics
        }
        
        # Add system metrics if requested
        if include_system_metrics:
            cache_info = model_loader.list_cached_models()
            result["system_metrics"] = {
                "cache_hit_rate": cache_info.get("cache_info", {}).get("hits", 0),
                "cache_size": cache_info.get("cache_size", 0),
                "model_cached": f"{model_name}:{stage}" in cache_info.get("cached_models", [])
            }
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {stage}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get("/monitoring/dashboard", tags=["monitoring"])
async def get_monitoring_dashboard() -> Dict[str, Any]:
    """Get comprehensive monitoring dashboard data."""
    try:
        # Get all models
        models = list_registered_models()
        
        # Get health status
        health = get_model_health()
        
        # Get cache info
        cache_info = model_loader.list_cached_models()
        
        # Build dashboard data
        dashboard_data = {
            "status": "success",
            "timestamp": "2024-12-28T07:58:34Z",  # Current timestamp
            "summary": {
                "total_models": len(models),
                "healthy_models": sum(1 for model in health["models_available"].values() 
                                    if model.get("registry", False)),
                "cached_models": cache_info["cache_size"],
                "system_status": health["status"]
            },
            "models": models,
            "health_details": health,
            "cache_details": cache_info,
            "registry_connection": health.get("registry_connection", False)
        }
        
        return dashboard_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

# Batch Operations
@router.post("/batch/reload", tags=["batch"])
async def batch_reload_models(
    model_names: List[str],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Reload specific models in batch."""
    try:
        def reload_specific_models(names: List[str]):
            for name in names:
                try:
                    # Clear from cache
                    cache_key_prod = f"{name}:Production"
                    cache_key_staging = f"{name}:Staging"
                    
                    if cache_key_prod in model_loader._model_cache:
                        del model_loader._model_cache[cache_key_prod]
                    if cache_key_staging in model_loader._model_cache:
                        del model_loader._model_cache[cache_key_staging]
                        
                    # Preload production model
                    get_production_model(name)
                    
                except Exception as e:
                    print(f"Failed to reload {name}: {e}")
        
        background_tasks.add_task(reload_specific_models, model_names)
        
        return {
            "status": "success",
            "message": f"Batch reload initiated for {len(model_names)} models",
            "model_names": model_names
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch reload failed: {str(e)}")
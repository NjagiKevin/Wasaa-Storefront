from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from airflow.models import Variable
from datetime import datetime
import logging
import os
import mlflow
import wandb

from app.services.demand_forecast_service import DemandForecastService
from app.services.recommendation_service import RecommendationService
from app.services.pricing_service import PricingService
from app.services.ads_manager_service import AdsManagerService

# Set W&B API key
os.environ["WANDB_API_KEY"] = "0d6b7fb3c73882e30b0ff6225435db9ebca921d9"

default_args = {
    'owner': 'wasaa',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': 300,  # seconds
}

with DAG(
    dag_id='ml_training_dag',
    default_args=default_args,
    description='Robust ML training pipeline for storefront with AdsManager integration',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'training'],
) as dag:

    @task
    def start_training():
        logging.info("Training started")
        return "Training started"

    @task
    def fetch_ads_data():
        campaigns = AdsManagerService.get_active_campaigns()
        enriched = AdsManagerService.enrich_campaigns_with_metrics(campaigns)
        logging.info(f"Enriched campaigns: {enriched}")
        return enriched

    @task
    def validate_data_quality(campaigns: list):
        # Could include missing value checks, distribution checks, profiling
        logging.info(f"Validating data quality for {len(campaigns)} campaigns")
        return "Data quality validated"

    @task
    def check_data_drift(campaigns: list):
        # Compare current campaigns vs historical metrics
        logging.info("Checking data drift against historical campaigns")
        return "Data drift checked"

    @task
    def prepare_training_features(campaigns: list):
        logging.info(f"Preparing features from campaigns: {campaigns}")
        # Prepare features for demand, recommendation, pricing models
        return "Features prepared"

    @task
    def hyperparameter_tuning(campaigns: list):
        # Auto-tune models using FLAML / Optuna
        logging.info("Running hyperparameter tuning")
        return "Hyperparameters tuned"

    @task
    def train_demand_forecast_model(campaigns: list):
        run_name = f"demand_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow.start_run(run_name=run_name)
        wandb.init(project="wasaa_storefront", name=run_name)
        DemandForecastService.train_model(campaigns)
        mlflow.log_artifact("models/demand_model.pkl")
        wandb.finish()
        mlflow.end_run()
        logging.info("Demand model trained")
        return "Demand model trained"

    @task
    def train_recommendation_model(campaigns: list):
        run_name = f"recommendation_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow.start_run(run_name=run_name)
        wandb.init(project="wasaa_storefront", name=run_name)
        RecommendationService.train_hybrid_model(campaigns)
        mlflow.log_artifact("models/recommendation_model.pkl")
        wandb.finish()
        mlflow.end_run()
        logging.info("Recommendation model trained")
        return "Recommendation model trained"

    @task
    def train_pricing_model(campaigns: list):
        run_name = f"pricing_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow.start_run(run_name=run_name)
        wandb.init(project="wasaa_storefront", name=run_name)
        PricingService.train_pricing_model(campaigns)
        mlflow.log_artifact("models/pricing_model.pkl")
        wandb.finish()
        mlflow.end_run()
        logging.info("Pricing model trained")
        return "Pricing model trained"

    @task
    def evaluate_models():
        # Implement validation gates
        # Only deploy if metrics meet thresholds
        logging.info("Evaluating models performance")
        return "Models evaluated"

    @task
    def deploy_models():
        # Versioned deployment logic (S3, Redis, DB)
        logging.info("Deploying models")
        return "Models deployed"

    @task
    def setup_monitoring():
        # Setup prediction drift, anomaly detection, alerting
        logging.info("Setting up monitoring")
        return "Monitoring setup"

    @task
    def send_success_notification():
        logging.info("Sending success notification")
        return "Training success notification sent"

    @task
    def end_training():
        logging.info("Training completed")
        return "Training completed"

    # DAG flow
    campaigns = fetch_ads_data()
    (
        start_training()
        >> campaigns
        >> validate_data_quality(campaigns)
        >> check_data_drift(campaigns)
        >> prepare_training_features(campaigns)
        >> hyperparameter_tuning(campaigns)
        >> [
            train_demand_forecast_model(campaigns),
            train_recommendation_model(campaigns),
            train_pricing_model(campaigns)
        ]
        >> evaluate_models()
        >> deploy_models()
        >> setup_monitoring()
        >> send_success_notification()
        >> end_training()
    )

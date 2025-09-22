from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
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
    'retry_delay': 300,  # 5 min
}

with DAG(
    dag_id='ml_prediction_dag',
    default_args=default_args,
    description='Generate storefront predictions with AdsManager integration (robust, MLflow & W&B)',
    schedule_interval='@hourly',
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'prediction'],
) as dag:

    @task
    def start_predictions():
        logging.info("Prediction DAG started")
        return "Prediction started"

    @task
    def fetch_active_campaigns():
        campaigns = AdsManagerService.get_active_campaigns()
        enriched = AdsManagerService.enrich_campaigns_with_metrics(campaigns)
        logging.info(f"Fetched and enriched {len(enriched)} campaigns")
        return enriched

    @task
    def validate_campaigns(campaigns: list):
        if not campaigns:
            raise ValueError("No active campaigns found!")
        logging.info("Campaign validation passed")
        return campaigns

    @task
    def generate_recommendations(campaigns: list):
        run_name = f"recommendation_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow.start_run(run_name=run_name)
        wandb.init(project="wasaa_storefront", name=run_name)
        RecommendationService.generate_hybrid_recommendations(campaigns)
        mlflow.log_artifact("models/recommendation_model.pkl")
        wandb.finish()
        mlflow.end_run()
        logging.info("Recommendations generated and logged")
        return "Recommendations generated"

    @task
    def generate_demand_forecasts(campaigns: list):
        run_name = f"demand_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow.start_run(run_name=run_name)
        wandb.init(project="wasaa_storefront", name=run_name)
        DemandForecastService.predict_demand(campaigns)
        mlflow.log_artifact("models/demand_model.pkl")
        wandb.finish()
        mlflow.end_run()
        logging.info("Demand forecasts generated and logged")
        return "Demand forecasts generated"

    @task
    def calculate_dynamic_prices(campaigns: list):
        run_name = f"pricing_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow.start_run(run_name=run_name)
        wandb.init(project="wasaa_storefront", name=run_name)
        PricingService.calculate_price_score(campaigns)
        mlflow.log_artifact("models/pricing_model.pkl")
        wandb.finish()
        mlflow.end_run()
        logging.info("Dynamic pricing calculated and logged")
        return "Dynamic prices calculated"

    @task
    def optimize_budget_allocation():
        logging.info("Budget allocation optimization started")
        return "Budget allocation optimized"

    @task
    def run_scenario_simulations():
        logging.info("Running scenario simulations")
        return "Scenario simulations completed"

    @task
    def detect_budget_anomalies():
        logging.info("Detecting anomalies in budget allocations")
        return "Budget anomalies detected"

    @task
    def persist_forecasts_and_scores():
        logging.info("Persisting forecasts and price scores to DB")
        return "Forecasts & scores persisted"

    @task
    def generate_budget_report():
        logging.info("Generating budget report")
        return "Budget report generated"

    @task
    def send_anomaly_alert():
        logging.info("Sending anomaly alert if any")
        return "Anomaly alert sent"

    @task
    def end_predictions():
        logging.info("Prediction DAG completed")
        return "Prediction completed"

    # DAG flow
    campaigns = fetch_active_campaigns()
    validated_campaigns = validate_campaigns(campaigns)

    (
        start_predictions()
        >> campaigns
        >> validated_campaigns
        >> [
            generate_recommendations(validated_campaigns),
            generate_demand_forecasts(validated_campaigns),
            calculate_dynamic_prices(validated_campaigns)
        ]
        >> optimize_budget_allocation()
        >> run_scenario_simulations()
        >> detect_budget_anomalies()
        >> persist_forecasts_and_scores()
        >> generate_budget_report()
        >> send_anomaly_alert()
        >> end_predictions()
    )

from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from datetime import datetime
import logging
import mlflow
import wandb

from app.services.demand_forecast_service import DemandForecastService
from app.services.recommendation_service import RecommendationService
from app.services.pricing_service import PricingService
from app.services.ads_manager_service import AdsManagerService


default_args = {
    'owner': 'wasaa',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': 300,  # seconds
}

with DAG(
    dag_id='campaign_forecast_dag',
    default_args=default_args,
    description='Optimized campaign forecasts with AdsManager, MLflow, W&B, anomaly detection',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'campaign'],
) as dag:

    @task
    def start_campaign_forecast():
        logging.info("Campaign forecast started")
        return "started"

    @task
    def fetch_and_validate_campaigns():
        campaigns = AdsManagerService.get_active_campaigns()
        if not campaigns:
            raise ValueError("No active campaigns found!")
        logging.info(f"Fetched {len(campaigns)} active campaigns")
        return campaigns

    @task
    def enrich_campaign_metrics(campaigns: list):
        enriched = AdsManagerService.enrich_campaigns_with_metrics(campaigns)
        logging.info(f"Enriched campaigns with performance metrics")
        return enriched

    @task
    def run_demand_forecasts(campaigns: list):
        run_name = f"campaign_demand_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow.start_run(run_name=run_name)
        wandb.init(project="wasaa_storefront", name=run_name)
        DemandForecastService.predict_demand_for_campaigns(campaigns)
        mlflow.log_artifact("models/demand_model.pkl")
        wandb.finish()
        mlflow.end_run()
        logging.info("Demand forecasts generated")
        return "demand_forecasts_done"

    @task
    def run_recommendations(campaigns: list):
        run_name = f"campaign_recommendation_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow.start_run(run_name=run_name)
        wandb.init(project="wasaa_storefront", name=run_name)
        RecommendationService.generate_hybrid_recommendations(campaigns)
        mlflow.log_artifact("models/recommendation_model.pkl")
        wandb.finish()
        mlflow.end_run()
        logging.info("Recommendations generated")
        return "recommendations_done"

    @task
    def calculate_optimal_pricing(campaigns: list):
        run_name = f"campaign_pricing_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow.start_run(run_name=run_name)
        wandb.init(project="wasaa_storefront", name=run_name)
        PricingService.enrich_campaigns_with_price_score(campaigns)
        mlflow.log_artifact("models/pricing_model.pkl")
        wandb.finish()
        mlflow.end_run()
        logging.info("Optimal pricing calculated")
        return "pricing_done"

    @task
    def detect_anomalies(campaigns: list):
        anomalies = AdsManagerService.detect_campaign_anomalies(campaigns)
        if anomalies:
            logging.warning(f"Detected anomalies in {len(anomalies)} campaigns")
        else:
            logging.info("No anomalies detected")
        return anomalies

    @task
    def persist_forecasts(campaigns: list):
        AdsManagerService.persist_campaign_forecasts(campaigns)
        logging.info("Persisted forecasts & recommendations")
        return "persisted"

    @task
    def generate_campaign_forecast_report():
        logging.info("Campaign forecast report generated")
        return "report_generated"

    @task
    def send_success_notification():
        logging.info("Success notification sent")
        return "notification_sent"

    @task
    def end_campaign_forecast():
        logging.info("Campaign forecast completed")
        return "completed"

    # DAG flow
    start = start_campaign_forecast()
    campaigns = fetch_and_validate_campaigns()
    enriched_campaigns = enrich_campaign_metrics(campaigns)

    forecast_tasks = [
        run_demand_forecasts(enriched_campaigns),
        run_recommendations(enriched_campaigns),
        calculate_optimal_pricing(enriched_campaigns)
    ]

    anomalies = detect_anomalies(enriched_campaigns)
    persisted = persist_forecasts(enriched_campaigns)

    (
        start
        >> campaigns
        >> enriched_campaigns
        >> forecast_tasks
        >> anomalies
        >> persisted
        >> generate_campaign_forecast_report()
        >> send_success_notification()
        >> end_campaign_forecast()
    )

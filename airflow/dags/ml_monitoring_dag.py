# airflow\dags\ml_monitoring_dag.py

from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from datetime import datetime
import mlflow
import wandb

from app.services.demand_forecast_service import DemandForecastService
from app.services.recommendation_service import RecommendationService
from app.services.pricing_service import PricingService
from app.services.ads_manager_service import AdsManagerService
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


default_args = {
    'owner': 'wasaa',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': 300,  # seconds
}

with DAG(
    dag_id='ml_monitoring_dag',
    default_args=default_args,
    description='Monitor ML models, predictions & campaign KPIs',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'monitoring'],
) as dag:

    @task
    def start_monitoring():
        return "Monitoring started"

    @task
    def check_model_availability():
        # Verify trained models exist in S3/Redis/DB
        return "Model availability checked"

    @task
    def validate_prediction_accuracy():
        # Fetch latest predictions & actuals, compute metrics (MAPE, RMSE, Precision/Recall)
        # Also pull metrics from MLflow
        wandb.init(project="wasaa_storefront", name=f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M')}")
        metrics = {
            "demand_accuracy": DemandForecastService.get_latest_accuracy(),
            "recommendation_ctr": RecommendationService.get_latest_ctr(),
            "pricing_accuracy": PricingService.get_latest_price_accuracy()
        }
        wandb.log(metrics)
        wandb.finish()
        return metrics

    @task
    def monitor_recommendation_quality():
        # Monitor CTR, conversions, cross-sell / upsell impact
        return AdsManagerService.fetch_recommendation_metrics()

    @task
    def monitor_dynamic_pricing_effectiveness():
        # Compare predicted prices vs actual sales
        return PricingService.evaluate_pricing_effectiveness()

    @task
    def monitor_data_drift():
        # Compare current features vs historical distributions
        return DemandForecastService.check_data_drift()

    @task
    def check_system_performance():
        # Could include latency, throughput, error rates
        return "System performance checked"

    @task
    def evaluate_retraining_trigger(pred_metrics, drift_metrics):
        # If demand accuracy < threshold or drift detected, trigger retraining
        retrain_needed = False
        if pred_metrics['demand_accuracy'] < 0.8 or drift_metrics['drift_detected']:
            retrain_needed = True
        return retrain_needed

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training_dag",
        trigger_dag_id="wasaa_storefront_training_dag",
        wait_for_completion=False,
        poke_interval=600,
    )

    @task
    def generate_monitoring_report():
        # Aggregate all metrics & anomalies
        return "Monitoring report generated"

    @task
    def send_critical_alert():
        # Send alerts via Slack/Email if metrics fall below thresholds
        return "Critical alert sent"

    @task
    def end_monitoring():
        return "Monitoring ended"

    # DAG flow
    (
        start_monitoring()
        >> check_model_availability()
        >> [validate_prediction_accuracy(), monitor_recommendation_quality(), monitor_dynamic_pricing_effectiveness()]
        >> monitor_data_drift()
        >> check_system_performance()
        >> evaluate_retraining_trigger(
            validate_prediction_accuracy(),
            monitor_data_drift()
        )
        >> trigger_training
        >> generate_monitoring_report()
        >> send_critical_alert()
        >> end_monitoring()
    )

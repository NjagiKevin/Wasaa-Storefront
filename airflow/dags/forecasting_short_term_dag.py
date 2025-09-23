from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
import logging

from app.services.demand_forecast_service import DemandForecastService

with DAG(
    dag_id='forecasting_short_term_dag',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'forecasting']
) as dag:

    @task
    def run_short_term():
        logging.info("Running short-term forecasts (7d)")
        DemandForecastService.predict_demand(["P001", "P002"])  # stub
        return "ok"

    run_short_term()
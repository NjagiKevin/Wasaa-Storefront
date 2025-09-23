from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
import logging

from app.services.demand_forecast_service import DemandForecastService

with DAG(
    dag_id='forecasting_long_term_dag',
    schedule_interval='0 6 * * 1',  # weekly at 6AM
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'forecasting']
) as dag:

    @task
    def run_long_term():
        logging.info("Running long-term forecasts (seasonal)")
        DemandForecastService.predict_demand(["P010", "P020"])  # stub
        return "ok"

    run_long_term()
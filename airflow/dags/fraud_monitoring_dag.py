from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago

from app.services.fraud_service import FraudService

with DAG(
    dag_id='fraud_monitoring_dag',
    schedule_interval='@hourly',
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'fraud']
) as dag:

    @task
    def monitor():
        return FraudService.latest_metrics()

    monitor()
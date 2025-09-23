from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from datetime import datetime
import logging

from app.services.fraud_service import FraudService


default_args = {
    'owner': 'wasaa',
    'depends_on_past': False,
}

with DAG(
    dag_id='fraud_training_dag',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'fraud']
) as dag:

    @task
    def train():
        logging.info("Training fraud model")
        FraudService.train_model([])
        return "trained"

    train()
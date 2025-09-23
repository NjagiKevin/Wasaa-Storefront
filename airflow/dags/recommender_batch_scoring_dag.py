from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from datetime import datetime
import logging

from app.services.recommendation_service import RecommendationService


default_args = {
    'owner': 'wasaa',
    'depends_on_past': False,
    'retries': 1,
}

with DAG(
    dag_id='recommender_batch_scoring_dag',
    default_args=default_args,
    schedule_interval='@hourly',
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'recommender']
) as dag:

    @task
    def batch_score():
        logging.info("Generating batch recommendations")
        try:
            RecommendationService.generate_hybrid_recommendations([])
        except Exception:
            logging.info("Batch scoring stub executed")
        return "scored"

    batch_score()
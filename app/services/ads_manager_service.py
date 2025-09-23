import requests
import logging
from typing import List, Dict

class AdsManagerClient:
    BASE_URL = "https://adsmanager.example.com/api"
    API_KEY = "your_api_key_here"

    @staticmethod
    def fetch_active_campaigns() -> List[Dict]:
        response = requests.get(f"{AdsManagerClient.BASE_URL}/campaigns/active",
                                headers={"Authorization": f"Bearer {AdsManagerClient.API_KEY}"})
        response.raise_for_status()
        campaigns = response.json()
        logging.info(f"Fetched {len(campaigns)} campaigns from AdsManager API")
        return campaigns

    @staticmethod
    def fetch_campaign_metrics(campaign_id: str) -> Dict:
        response = requests.get(f"{AdsManagerClient.BASE_URL}/campaigns/{campaign_id}/metrics",
                                headers={"Authorization": f"Bearer {AdsManagerClient.API_KEY}"})
        response.raise_for_status()
        metrics = response.json()
        logging.info(f"Fetched metrics for campaign {campaign_id}")
        return metrics

class AdsManagerService:
    """
    Service wrapper exposing methods used in Airflow DAGs.
    """
    @staticmethod
    def get_active_campaigns() -> List[Dict]:
        try:
            return AdsManagerClient.fetch_active_campaigns()
        except Exception:
            logging.warning("Falling back to dummy campaigns")
            return [{"campaign_id": "C001", "product_id": "P001", "status": "active"}]

    @staticmethod
    def enrich_campaigns_with_metrics(campaigns: List[Dict]) -> List[Dict]:
        enriched = []
        for c in campaigns:
            try:
                metrics = AdsManagerClient.fetch_campaign_metrics(c.get("campaign_id", ""))
            except Exception:
                metrics = {"ctr": 0.02, "cvr": 0.01, "spend": 100}
            e = {**c, **metrics}
            enriched.append(e)
        return enriched

    @staticmethod
    def detect_campaign_anomalies(campaigns: List[Dict]) -> List[Dict]:
        anomalies = []
        for c in campaigns:
            if c.get("ctr", 0) < 0.005:  # simple rule
                anomalies.append({"campaign_id": c.get("campaign_id"), "reason": "low_ctr"})
        return anomalies

    @staticmethod
    def persist_campaign_forecasts(campaigns: List[Dict]) -> None:
        logging.info(f"Persisting {len(campaigns)} campaign forecasts (stub)")

    @staticmethod
    def fetch_recommendation_metrics() -> dict:
        return {"ctr_7d": 0.07, "conversion_7d": 0.012}

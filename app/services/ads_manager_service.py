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

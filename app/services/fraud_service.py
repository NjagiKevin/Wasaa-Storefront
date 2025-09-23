from typing import Dict, Any, List
import logging
import numpy as np

class FraudService:
    """
    Simple fraud scoring service stub. Replace with GBM/Isolation Forest.
    """
    @staticmethod
    def train_model(events: List[Dict[str, Any]]) -> None:
        logging.info(f"Training fraud model on {len(events)} events (stub)")

    @staticmethod
    def score_event(event: Dict[str, Any]) -> float:
        # Dummy score based on simple heuristics
        amount = float(event.get('amount', 0))
        return float(min(0.99, max(0.01, (amount % 100) / 100.0)))

    @staticmethod
    def latest_metrics() -> Dict[str, float]:
        return {"auc": 0.85, "precision": 0.8, "recall": 0.7}
"""
Advanced Fraud Detection with GBM, Random Forest, Isolation Forest
Includes real-time anomaly detection and adaptive risk levels
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import logging
import mlflow
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ML models
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Feature engineering
from sklearn.feature_selection import SelectKBest, f_classif
import shap

# Real-time processing
import threading
from collections import deque
import time

class FeatureEngineer:
    """
    Advanced feature engineering for fraud detection
    """
    
    def __init__(self):
        self.user_behavior_cache = {}
        self.merchant_stats_cache = {}
        self.transaction_patterns_cache = deque(maxlen=10000)
    
    def extract_user_behavior_features(self, user_id: str, transaction_data: Dict) -> Dict[str, float]:
        """Extract user behavior features"""
        
        current_time = datetime.now()
        
        # Initialize user cache if not exists
        if user_id not in self.user_behavior_cache:
            self.user_behavior_cache[user_id] = {
                'transaction_count': 0,
                'total_amount': 0,
                'last_transaction_time': None,
                'average_amount': 0,
                'std_amount': 0,
                'transaction_amounts': deque(maxlen=100),
                'transaction_times': deque(maxlen=100),
                'unique_merchants': set(),
                'failed_attempts': 0,
                'success_rate': 1.0
            }
        
        user_cache = self.user_behavior_cache[user_id]
        
        # Current transaction amount
        amount = float(transaction_data.get('amount', 0))
        
        # Update cache
        user_cache['transaction_count'] += 1
        user_cache['total_amount'] += amount
        user_cache['transaction_amounts'].append(amount)
        user_cache['transaction_times'].append(current_time)
        user_cache['unique_merchants'].add(transaction_data.get('merchant_id', 'unknown'))
        
        # Calculate features
        features = {}
        
        # Amount-based features
        if len(user_cache['transaction_amounts']) > 1:
            amounts = list(user_cache['transaction_amounts'])
            user_cache['average_amount'] = np.mean(amounts)
            user_cache['std_amount'] = np.std(amounts)
            
            features['amount_vs_average'] = amount / max(user_cache['average_amount'], 1)
            features['amount_z_score'] = (amount - user_cache['average_amount']) / max(user_cache['std_amount'], 1)
        else:
            features['amount_vs_average'] = 1.0
            features['amount_z_score'] = 0.0
        
        # Time-based features
        if user_cache['last_transaction_time']:
            time_diff = (current_time - user_cache['last_transaction_time']).total_seconds()
            features['time_since_last_transaction'] = min(time_diff / 3600, 24)  # Hours, capped at 24
        else:
            features['time_since_last_transaction'] = 24
        
        user_cache['last_transaction_time'] = current_time
        
        # Velocity features (last hour, last day)
        hour_ago = current_time - timedelta(hours=1)
        day_ago = current_time - timedelta(days=1)
        
        recent_transactions_hour = [t for t in user_cache['transaction_times'] if t >= hour_ago]
        recent_transactions_day = [t for t in user_cache['transaction_times'] if t >= day_ago]
        
        features['transactions_last_hour'] = len(recent_transactions_hour)
        features['transactions_last_day'] = len(recent_transactions_day)
        
        # Diversity features
        features['unique_merchants_count'] = len(user_cache['unique_merchants'])
        features['transaction_count_total'] = user_cache['transaction_count']
        features['success_rate'] = user_cache['success_rate']
        
        return features
    
    def extract_transaction_features(self, transaction_data: Dict) -> Dict[str, float]:
        """Extract transaction-specific features"""
        
        features = {}
        current_time = datetime.now()
        
        # Basic transaction features
        features['amount'] = float(transaction_data.get('amount', 0))
        features['amount_log'] = np.log1p(features['amount'])
        
        # Time features
        features['hour_of_day'] = current_time.hour / 24.0
        features['day_of_week'] = current_time.weekday() / 6.0
        features['is_weekend'] = float(current_time.weekday() >= 5)
        features['is_business_hours'] = float(9 <= current_time.hour <= 17)
        features['is_late_night'] = float(22 <= current_time.hour or current_time.hour <= 6)
        
        # Location features (if available)
        if 'location' in transaction_data:
            # Simplified location risk scoring
            location = transaction_data['location']
            features['location_risk_score'] = self.get_location_risk_score(location)
        else:
            features['location_risk_score'] = 0.5  # Neutral
        
        # Payment method features
        payment_method = transaction_data.get('payment_method', 'unknown')
        features['payment_method_risk'] = self.get_payment_method_risk(payment_method)
        
        # Merchant features
        merchant_id = transaction_data.get('merchant_id', 'unknown')
        features.update(self.get_merchant_features(merchant_id))
        
        return features
    
    def get_location_risk_score(self, location: str) -> float:
        """Calculate location-based risk score"""
        # Simplified risk scoring based on location
        high_risk_locations = ['unknown', 'international', 'high_crime_area']
        if any(risk_loc in location.lower() for risk_loc in high_risk_locations):
            return 0.8
        return 0.2
    
    def get_payment_method_risk(self, payment_method: str) -> float:
        """Calculate payment method risk score"""
        risk_scores = {
            'credit_card': 0.3,
            'debit_card': 0.2,
            'mobile_money': 0.4,
            'bank_transfer': 0.1,
            'cash': 0.1,
            'crypto': 0.8,
            'unknown': 0.7
        }
        return risk_scores.get(payment_method.lower(), 0.5)
    
    def get_merchant_features(self, merchant_id: str) -> Dict[str, float]:
        """Extract merchant-specific features"""
        
        if merchant_id not in self.merchant_stats_cache:
            self.merchant_stats_cache[merchant_id] = {
                'transaction_count': 0,
                'fraud_count': 0,
                'total_amount': 0,
                'fraud_rate': 0.0,
                'risk_score': 0.5
            }
        
        merchant_stats = self.merchant_stats_cache[merchant_id]
        
        return {
            'merchant_fraud_rate': merchant_stats['fraud_rate'],
            'merchant_transaction_count': min(merchant_stats['transaction_count'] / 1000, 1.0),
            'merchant_risk_score': merchant_stats['risk_score']
        }
    
    def update_merchant_stats(self, merchant_id: str, is_fraud: bool):
        """Update merchant statistics after fraud confirmation"""
        
        if merchant_id in self.merchant_stats_cache:
            stats = self.merchant_stats_cache[merchant_id]
            stats['transaction_count'] += 1
            if is_fraud:
                stats['fraud_count'] += 1
            stats['fraud_rate'] = stats['fraud_count'] / stats['transaction_count']
            stats['risk_score'] = min(stats['fraud_rate'] * 2, 1.0)

class AdvancedFraudDetector:
    """
    Advanced fraud detection system with multiple ML models
    """
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.feature_scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Adaptive risk thresholds
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.95
        }
        
        # Real-time monitoring
        self.recent_predictions = deque(maxlen=1000)
        self.model_performance = {}
        
    def prepare_features(self, transaction_data: Dict) -> pd.DataFrame:
        """Prepare comprehensive feature set for a transaction"""
        
        user_id = transaction_data.get('user_id', 'unknown')
        
        # Extract all feature types
        user_features = self.feature_engineer.extract_user_behavior_features(user_id, transaction_data)
        transaction_features = self.feature_engineer.extract_transaction_features(transaction_data)
        
        # Combine features
        all_features = {**user_features, **transaction_features}
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([all_features])
        
        return feature_df
    
    def train_gradient_boosting_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
        """Train Gradient Boosting Classifier"""
        
        with mlflow.start_run(run_name="fraud_gradient_boosting"):
            
            # Handle class imbalance with SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            
            # Grid search for best parameters
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.05],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9]
            }
            
            gb_model = GradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(gb_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_balanced, y_balanced)
            
            best_model = grid_search.best_estimator_
            
            # Log parameters and metrics
            mlflow.log_params(grid_search.best_params_)
            
            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_balanced, y_balanced, cv=5, scoring='roc_auc')
            mlflow.log_metric("cv_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_auc_std", cv_scores.std())
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
            
        return best_model
    
    def train_random_forest_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """Train Random Forest Classifier"""
        
        with mlflow.start_run(run_name="fraud_random_forest"):
            
            # Handle class imbalance
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            
            # Grid search
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            }
            
            rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_balanced, y_balanced)
            
            best_model = grid_search.best_estimator_
            
            # Log results
            mlflow.log_params(grid_search.best_params_)
            
            cv_scores = cross_val_score(best_model, X_balanced, y_balanced, cv=5, scoring='roc_auc')
            mlflow.log_metric("cv_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_auc_std", cv_scores.std())
            
        return best_model
    
    def train_catboost_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> CatBoostClassifier:
        """Train CatBoost Classifier"""
        
        with mlflow.start_run(run_name="fraud_catboost"):
            
            # CatBoost handles imbalanced data well with class_weights
            catboost_model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.1,
                depth=6,
                class_weights=[1, 5],  # Give more weight to fraud class
                random_seed=42,
                verbose=False
            )
            
            catboost_model.fit(X_train, y_train)
            
            # Log parameters
            mlflow.log_param("iterations", 500)
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_param("depth", 6)
            
            # Cross-validation
            cv_scores = cross_val_score(catboost_model, X_train, y_train, cv=5, scoring='roc_auc')
            mlflow.log_metric("cv_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_auc_std", cv_scores.std())
            
        return catboost_model
    
    def train_isolation_forest(self, X_train: pd.DataFrame) -> IsolationForest:
        """Train Isolation Forest for anomaly detection"""
        
        with mlflow.start_run(run_name="fraud_isolation_forest"):
            
            # Only train on normal transactions (unsupervised)
            isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            isolation_forest.fit(X_train)
            
            # Log parameters
            mlflow.log_param("contamination", 0.1)
            mlflow.log_param("model_type", "isolation_forest")
            
        return isolation_forest
    
    def train_all_models(self, training_data: pd.DataFrame):
        """Train all fraud detection models"""
        
        # Prepare features for all transactions
        feature_list = []
        labels = []
        
        for idx, row in training_data.iterrows():
            transaction_data = row.to_dict()
            features = self.prepare_features(transaction_data)
            feature_list.append(features.iloc[0])
            labels.append(row.get('is_fraud', 0))
        
        # Convert to DataFrame
        X = pd.DataFrame(feature_list)
        y = pd.Series(labels)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.feature_scaler.fit_transform(X),
            columns=X.columns
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train supervised models
        try:
            self.models['gradient_boosting'] = self.train_gradient_boosting_model(X_train, y_train)
            logging.info("Gradient Boosting model trained successfully")
        except Exception as e:
            logging.error(f"Gradient Boosting training failed: {e}")
        
        try:
            self.models['random_forest'] = self.train_random_forest_model(X_train, y_train)
            logging.info("Random Forest model trained successfully")
        except Exception as e:
            logging.error(f"Random Forest training failed: {e}")
        
        try:
            self.models['catboost'] = self.train_catboost_model(X_train, y_train)
            logging.info("CatBoost model trained successfully")
        except Exception as e:
            logging.error(f"CatBoost training failed: {e}")
        
        # Train unsupervised model (only on normal transactions)
        try:
            normal_transactions = X_train[y_train == 0]
            self.models['isolation_forest'] = self.train_isolation_forest(normal_transactions)
            logging.info("Isolation Forest model trained successfully")
        except Exception as e:
            logging.error(f"Isolation Forest training failed: {e}")
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate all trained models"""
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'isolation_forest':
                    # Anomaly detection evaluation
                    predictions = model.predict(X_test)
                    # Convert anomaly scores: -1 (anomaly) -> 1 (fraud), 1 (normal) -> 0 (not fraud)
                    fraud_predictions = [1 if pred == -1 else 0 for pred in predictions]
                else:
                    # Classification evaluation
                    fraud_predictions = model.predict(X_test)
                
                # Calculate metrics
                auc_score = roc_auc_score(y_test, fraud_predictions)
                
                self.model_performance[model_name] = {
                    'auc_score': auc_score,
                    'last_updated': datetime.now()
                }
                
                logging.info(f"{model_name} - AUC: {auc_score:.4f}")
                
            except Exception as e:
                logging.error(f"Evaluation failed for {model_name}: {e}")
    
    def predict_fraud_probability(self, transaction_data: Dict) -> Dict[str, Union[float, str]]:
        """Predict fraud probability for a single transaction"""
        
        # Prepare features
        features = self.prepare_features(transaction_data)
        features_scaled = pd.DataFrame(
            self.feature_scaler.transform(features),
            columns=features.columns
        )
        
        predictions = {}
        ensemble_score = 0.0
        model_count = 0
        
        # Get predictions from all models
        for model_name, model in self.models.items():
            try:
                if model_name == 'isolation_forest':
                    # Anomaly score
                    anomaly_score = model.decision_function(features_scaled)[0]
                    # Convert to probability (higher is more fraudulent)
                    fraud_prob = max(0, -anomaly_score) / 2.0  # Normalize
                else:
                    # Classification probability
                    if hasattr(model, 'predict_proba'):
                        fraud_prob = model.predict_proba(features_scaled)[0][1]
                    else:
                        fraud_prob = model.decision_function(features_scaled)[0]
                        fraud_prob = 1 / (1 + np.exp(-fraud_prob))  # Sigmoid
                
                predictions[model_name] = fraud_prob
                ensemble_score += fraud_prob
                model_count += 1
                
            except Exception as e:
                logging.warning(f"Prediction failed for {model_name}: {e}")
                predictions[model_name] = 0.5  # Default neutral score
        
        # Calculate ensemble score
        if model_count > 0:
            ensemble_score /= model_count
        else:
            ensemble_score = 0.5
        
        # Determine risk level
        risk_level = self.get_risk_level(ensemble_score)
        
        # Store prediction for monitoring
        self.recent_predictions.append({
            'timestamp': datetime.now(),
            'user_id': transaction_data.get('user_id', 'unknown'),
            'amount': transaction_data.get('amount', 0),
            'fraud_probability': ensemble_score,
            'risk_level': risk_level
        })
        
        return {
            'fraud_probability': ensemble_score,
            'risk_level': risk_level,
            'individual_predictions': predictions,
            'requires_2fa': self.requires_2fa(ensemble_score, transaction_data),
            'requires_manual_review': self.requires_manual_review(ensemble_score, transaction_data),
            'confidence': min(model_count / len(self.models), 1.0) if self.models else 0.0
        }
    
    def get_risk_level(self, fraud_probability: float) -> str:
        """Determine risk level based on fraud probability"""
        
        if fraud_probability >= self.risk_thresholds['critical']:
            return 'critical'
        elif fraud_probability >= self.risk_thresholds['high']:
            return 'high'
        elif fraud_probability >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def requires_2fa(self, fraud_probability: float, transaction_data: Dict) -> bool:
        """Determine if transaction requires 2FA"""
        
        # High risk transactions require 2FA
        if fraud_probability >= self.risk_thresholds['medium']:
            return True
        
        # Large amounts require 2FA regardless of risk
        amount = float(transaction_data.get('amount', 0))
        if amount > 10000:  # Adjust threshold as needed
            return True
        
        return False
    
    def requires_manual_review(self, fraud_probability: float, transaction_data: Dict) -> bool:
        """Determine if transaction requires manual review"""
        
        # Critical risk always requires manual review
        if fraud_probability >= self.risk_thresholds['high']:
            return True
        
        # Very large amounts require manual review
        amount = float(transaction_data.get('amount', 0))
        if amount > 50000:  # Adjust threshold as needed
            return True
        
        return False
    
    def update_adaptive_thresholds(self):
        """Update risk thresholds based on recent performance"""
        
        if len(self.recent_predictions) < 100:
            return
        
        # Calculate recent fraud rates by risk level
        recent_df = pd.DataFrame(list(self.recent_predictions))
        
        # This would be implemented with actual fraud confirmations
        # For now, use placeholder logic
        
        # Example: If too many false positives, increase thresholds
        false_positive_rate = 0.05  # Would be calculated from actual data
        
        if false_positive_rate > 0.1:
            # Increase thresholds to reduce false positives
            for level in self.risk_thresholds:
                self.risk_thresholds[level] = min(self.risk_thresholds[level] * 1.1, 0.99)
        elif false_positive_rate < 0.02:
            # Decrease thresholds to catch more fraud
            for level in self.risk_thresholds:
                self.risk_thresholds[level] = max(self.risk_thresholds[level] * 0.9, 0.01)
        
        logging.info(f"Updated risk thresholds: {self.risk_thresholds}")
    
    def get_feature_importance(self, model_name: str = 'gradient_boosting') -> pd.DataFrame:
        """Get feature importance from specified model"""
        
        if model_name not in self.models:
            return pd.DataFrame()
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Get feature names (this would need to be stored during training)
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return pd.DataFrame()
    
    def explain_prediction(self, transaction_data: Dict, model_name: str = 'gradient_boosting') -> Dict:
        """Explain a fraud prediction using SHAP values"""
        
        if model_name not in self.models:
            return {}
        
        try:
            # Prepare features
            features = self.prepare_features(transaction_data)
            features_scaled = pd.DataFrame(
                self.feature_scaler.transform(features),
                columns=features.columns
            )
            
            model = self.models[model_name]
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_scaled)
            
            # Get SHAP values for fraud class (index 1)
            if len(shap_values) == 2:  # Binary classification
                fraud_shap_values = shap_values[1][0]
            else:
                fraud_shap_values = shap_values[0]
            
            # Create explanation
            explanation = {
                'base_value': explainer.expected_value[1] if len(shap_values) == 2 else explainer.expected_value,
                'shap_values': dict(zip(features.columns, fraud_shap_values)),
                'feature_values': dict(zip(features.columns, features.iloc[0].values))
            }
            
            return explanation
            
        except Exception as e:
            logging.error(f"SHAP explanation failed: {e}")
            return {}
    
    def get_real_time_monitoring_stats(self) -> Dict:
        """Get real-time monitoring statistics"""
        
        if not self.recent_predictions:
            return {}
        
        recent_df = pd.DataFrame(list(self.recent_predictions))
        
        # Calculate statistics for different time windows
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        hour_data = recent_df[recent_df['timestamp'] >= last_hour]
        day_data = recent_df[recent_df['timestamp'] >= last_day]
        
        stats = {
            'total_predictions': len(recent_df),
            'last_hour': {
                'predictions': len(hour_data),
                'avg_fraud_probability': hour_data['fraud_probability'].mean() if len(hour_data) > 0 else 0,
                'high_risk_count': len(hour_data[hour_data['risk_level'].isin(['high', 'critical'])]),
                'avg_amount': hour_data['amount'].mean() if len(hour_data) > 0 else 0
            },
            'last_day': {
                'predictions': len(day_data),
                'avg_fraud_probability': day_data['fraud_probability'].mean() if len(day_data) > 0 else 0,
                'high_risk_count': len(day_data[day_data['risk_level'].isin(['high', 'critical'])]),
                'avg_amount': day_data['amount'].mean() if len(day_data) > 0 else 0
            },
            'risk_level_distribution': recent_df['risk_level'].value_counts().to_dict(),
            'model_performance': self.model_performance
        }
        
        return stats
    
    def save_models(self, model_path: str):
        """Save all trained models"""
        
        models_data = {
            'models': self.models,
            'feature_scaler': self.feature_scaler,
            'risk_thresholds': self.risk_thresholds,
            'model_performance': self.model_performance
        }
        
        joblib.dump(models_data, f"{model_path}/advanced_fraud_models.pkl")
        
        logging.info(f"Fraud detection models saved to {model_path}")
    
    def load_models(self, model_path: str):
        """Load all trained models"""
        
        try:
            models_data = joblib.load(f"{model_path}/advanced_fraud_models.pkl")
            self.models = models_data.get('models', {})
            self.feature_scaler = models_data.get('feature_scaler', StandardScaler())
            self.risk_thresholds = models_data.get('risk_thresholds', {
                'low': 0.3, 'medium': 0.6, 'high': 0.8, 'critical': 0.95
            })
            self.model_performance = models_data.get('model_performance', {})
            
            logging.info(f"Fraud detection models loaded from {model_path}")
            
        except Exception as e:
            logging.error(f"Failed to load fraud detection models: {e}")

# Factory function
def create_advanced_fraud_detector() -> AdvancedFraudDetector:
    """Create and return an advanced fraud detector instance"""
    return AdvancedFraudDetector()

# Real-time monitoring thread
class FraudMonitoringThread(threading.Thread):
    """
    Background thread for real-time fraud monitoring and threshold adaptation
    """
    
    def __init__(self, fraud_detector: AdvancedFraudDetector, update_interval: int = 300):
        super().__init__(daemon=True)
        self.fraud_detector = fraud_detector
        self.update_interval = update_interval
        self.running = True
    
    def run(self):
        """Run the monitoring loop"""
        while self.running:
            try:
                # Update adaptive thresholds
                self.fraud_detector.update_adaptive_thresholds()
                
                # Log monitoring stats
                stats = self.fraud_detector.get_real_time_monitoring_stats()
                if stats:
                    logging.info(f"Fraud monitoring stats: {stats['last_hour']}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logging.error(f"Fraud monitoring thread error: {e}")
                time.sleep(60)  # Wait before retrying
    
    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
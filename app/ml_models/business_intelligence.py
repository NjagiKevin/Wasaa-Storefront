"""
Advanced Business Intelligence Features
- Customer Lifetime Value (CLV) Prediction
- Churn Prediction
- Advanced Product Bundling
- Merchant Dashboard Integration
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score
from catboost import CatBoostRegressor, CatBoostClassifier

# Clustering for segmentation
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

class CustomerLifetimeValuePredictor:
    """
    Predict Customer Lifetime Value using advanced ML models
    """
    
    def __init__(self):
        self.model = None
        self.feature_scaler = StandardScaler()
        self.clv_segments = {}
        
    def calculate_clv_features(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for CLV prediction"""
        
        features = pd.DataFrame()
        
        # Basic customer features
        features['customer_age_days'] = (datetime.now() - pd.to_datetime(customer_data['registration_date'])).dt.days
        features['total_orders'] = customer_data['total_orders']
        features['total_spent'] = customer_data['total_spent']
        features['average_order_value'] = customer_data['total_spent'] / customer_data['total_orders'].clip(lower=1)
        
        # Frequency features
        features['order_frequency'] = customer_data['total_orders'] / (features['customer_age_days'] / 30).clip(lower=1)  # Orders per month
        features['days_since_last_order'] = (datetime.now() - pd.to_datetime(customer_data['last_order_date'])).dt.days
        
        # Behavioral features
        features['unique_categories'] = customer_data.get('unique_categories_purchased', 1)
        features['return_rate'] = customer_data.get('return_rate', 0)
        features['review_score'] = customer_data.get('average_review_score', 4.0)
        features['support_tickets'] = customer_data.get('support_ticket_count', 0)
        
        # Geographic features
        features['is_urban'] = customer_data.get('is_urban_location', True).astype(float)
        features['distance_from_warehouse'] = customer_data.get('distance_from_warehouse', 50)
        
        # Engagement features
        features['email_open_rate'] = customer_data.get('email_open_rate', 0.3)
        features['app_usage_hours'] = customer_data.get('monthly_app_hours', 10)
        features['social_media_engagement'] = customer_data.get('social_engagement_score', 0.1)
        
        # Financial features
        features['payment_method_diversity'] = customer_data.get('unique_payment_methods', 1)
        features['failed_payment_rate'] = customer_data.get('failed_payment_rate', 0)
        features['uses_credit'] = customer_data.get('uses_credit_payment', False).astype(float)
        
        return features
    
    def train_clv_model(self, customer_data: pd.DataFrame, target_column: str = 'actual_clv'):
        """Train the CLV prediction model"""
        
        # Calculate features
        X = self.calculate_clv_features(customer_data)
        y = customer_data[target_column]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.feature_scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        with mlflow.start_run(run_name="clv_prediction"):
            
            # Train CatBoost model (handles mixed data types well)
            self.model = CatBoostRegressor(
                iterations=500,
                learning_rate=0.1,
                depth=6,
                random_seed=42,
                verbose=False
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_predictions = self.model.predict(X_train)
            test_predictions = self.model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            
            # Log metrics
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_param("model_type", "catboost_regressor")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            mlflow.log_text(feature_importance.to_string(), "clv_feature_importance.txt")
            
            logging.info(f"CLV model trained - Test RMSE: {test_rmse:.2f}")
            
        # Create CLV segments
        self._create_clv_segments(customer_data, y)
        
    def _create_clv_segments(self, customer_data: pd.DataFrame, clv_values: pd.Series):
        """Create customer segments based on CLV"""
        
        clv_percentiles = np.percentile(clv_values, [33, 66])
        
        self.clv_segments = {
            'high_value': clv_values[clv_values >= clv_percentiles[1]].index.tolist(),
            'medium_value': clv_values[(clv_values >= clv_percentiles[0]) & (clv_values < clv_percentiles[1])].index.tolist(),
            'low_value': clv_values[clv_values < clv_percentiles[0]].index.tolist()
        }
        
        logging.info(f"CLV segments created: {len(self.clv_segments['high_value'])} high, "
                    f"{len(self.clv_segments['medium_value'])} medium, "
                    f"{len(self.clv_segments['low_value'])} low value customers")
    
    def predict_clv(self, customer_data: Dict) -> Dict[str, Union[float, str]]:
        """Predict CLV for a single customer"""
        
        if self.model is None:
            return {'predicted_clv': 0, 'segment': 'unknown', 'confidence': 0}
        
        try:
            # Convert to DataFrame for feature calculation
            customer_df = pd.DataFrame([customer_data])
            
            # Calculate features
            features = self.calculate_clv_features(customer_df)
            features_scaled = pd.DataFrame(
                self.feature_scaler.transform(features),
                columns=features.columns
            )
            
            # Predict CLV
            predicted_clv = self.model.predict(features_scaled)[0]
            
            # Determine segment
            if predicted_clv >= 1000:  # High value threshold
                segment = 'high_value'
            elif predicted_clv >= 500:  # Medium value threshold
                segment = 'medium_value'
            else:
                segment = 'low_value'
            
            return {
                'predicted_clv': float(predicted_clv),
                'segment': segment,
                'confidence': 0.85,  # Would be calculated from model uncertainty
                'features_used': list(features.columns),
                'top_contributing_factors': self._get_top_factors(features.iloc[0])
            }
            
        except Exception as e:
            logging.error(f"CLV prediction failed: {e}")
            return {'predicted_clv': 0, 'segment': 'unknown', 'confidence': 0}
    
    def _get_top_factors(self, customer_features: pd.Series) -> List[Dict]:
        """Get top contributing factors for CLV"""
        
        if self.model is None:
            return []
        
        try:
            # Get feature importance
            importance_dict = dict(zip(customer_features.index, self.model.feature_importances_))
            
            # Get top 5 factors
            top_factors = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            
            factors = []
            for factor, importance in top_factors:
                factors.append({
                    'factor': factor,
                    'importance': float(importance),
                    'customer_value': float(customer_features[factor])
                })
            
            return factors
            
        except Exception as e:
            logging.error(f"Failed to get top factors: {e}")
            return []

class ChurnPredictor:
    """
    Predict customer churn using advanced ML models
    """
    
    def __init__(self):
        self.model = None
        self.feature_scaler = StandardScaler()
        self.churn_threshold = 0.5
        
    def calculate_churn_features(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for churn prediction"""
        
        features = pd.DataFrame()
        current_date = datetime.now()
        
        # Recency features
        features['days_since_last_order'] = (current_date - pd.to_datetime(customer_data['last_order_date'])).dt.days
        features['days_since_registration'] = (current_date - pd.to_datetime(customer_data['registration_date'])).dt.days
        features['days_since_last_login'] = (current_date - pd.to_datetime(customer_data.get('last_login_date', current_date))).dt.days
        
        # Frequency features
        features['total_orders'] = customer_data['total_orders']
        features['orders_per_month'] = customer_data['total_orders'] / (features['days_since_registration'] / 30).clip(lower=1)
        features['order_frequency_trend'] = customer_data.get('order_frequency_trend', 0)  # Increasing/decreasing trend
        
        # Monetary features
        features['total_spent'] = customer_data['total_spent']
        features['average_order_value'] = customer_data['total_spent'] / customer_data['total_orders'].clip(lower=1)
        features['spending_trend'] = customer_data.get('spending_trend', 0)  # Increasing/decreasing trend
        
        # Behavioral features
        features['support_tickets'] = customer_data.get('support_ticket_count', 0)
        features['return_rate'] = customer_data.get('return_rate', 0)
        features['review_sentiment'] = customer_data.get('average_review_sentiment', 0.5)
        features['app_usage_decline'] = customer_data.get('app_usage_decline', 0)
        
        # Engagement features
        features['email_open_rate'] = customer_data.get('email_open_rate', 0.3)
        features['email_open_rate_trend'] = customer_data.get('email_open_trend', 0)
        features['social_engagement'] = customer_data.get('social_engagement_score', 0.1)
        features['newsletter_subscriber'] = customer_data.get('newsletter_subscriber', False).astype(float)
        
        # Product interaction features
        features['category_diversity'] = customer_data.get('unique_categories_purchased', 1)
        features['favorite_category_dominance'] = customer_data.get('favorite_category_percentage', 0.5)
        features['new_product_adoption'] = customer_data.get('new_product_adoption_rate', 0.1)
        
        # Financial features
        features['failed_payments'] = customer_data.get('failed_payment_count', 0)
        features['payment_method_changes'] = customer_data.get('payment_method_changes', 0)
        features['discount_dependency'] = customer_data.get('discount_usage_rate', 0)
        
        return features
    
    def train_churn_model(self, customer_data: pd.DataFrame, target_column: str = 'is_churned'):
        """Train the churn prediction model"""
        
        # Calculate features
        X = self.calculate_churn_features(customer_data)
        y = customer_data[target_column].astype(int)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.feature_scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        with mlflow.start_run(run_name="churn_prediction"):
            
            # Train CatBoost classifier
            self.model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.1,
                depth=6,
                class_weights=[1, 3],  # Give more weight to churn class
                random_seed=42,
                verbose=False
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_predictions = self.model.predict_proba(X_train)[:, 1]
            test_predictions = self.model.predict_proba(X_test)[:, 1]
            
            train_auc = roc_auc_score(y_train, train_predictions)
            test_auc = roc_auc_score(y_test, test_predictions)
            
            # Log metrics
            mlflow.log_metric("train_auc", train_auc)
            mlflow.log_metric("test_auc", test_auc)
            mlflow.log_param("model_type", "catboost_classifier")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            mlflow.log_text(feature_importance.to_string(), "churn_feature_importance.txt")
            
            logging.info(f"Churn model trained - Test AUC: {test_auc:.3f}")
    
    def predict_churn_probability(self, customer_data: Dict) -> Dict[str, Union[float, str, List]]:
        """Predict churn probability for a single customer"""
        
        if self.model is None:
            return {'churn_probability': 0.5, 'risk_level': 'unknown', 'confidence': 0}
        
        try:
            # Convert to DataFrame for feature calculation
            customer_df = pd.DataFrame([customer_data])
            
            # Calculate features
            features = self.calculate_churn_features(customer_df)
            features_scaled = pd.DataFrame(
                self.feature_scaler.transform(features),
                columns=features.columns
            )
            
            # Predict churn probability
            churn_prob = self.model.predict_proba(features_scaled)[0][1]
            
            # Determine risk level
            if churn_prob >= 0.8:
                risk_level = 'critical'
            elif churn_prob >= 0.6:
                risk_level = 'high'
            elif churn_prob >= 0.4:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            return {
                'churn_probability': float(churn_prob),
                'risk_level': risk_level,
                'confidence': 0.85,
                'retention_recommendations': self._get_retention_recommendations(risk_level, features.iloc[0]),
                'key_risk_factors': self._get_key_risk_factors(features.iloc[0])
            }
            
        except Exception as e:
            logging.error(f"Churn prediction failed: {e}")
            return {'churn_probability': 0.5, 'risk_level': 'unknown', 'confidence': 0}
    
    def _get_retention_recommendations(self, risk_level: str, customer_features: pd.Series) -> List[str]:
        """Get retention recommendations based on risk level"""
        
        recommendations = []
        
        if risk_level in ['critical', 'high']:
            recommendations.extend([
                "Immediate outreach with personalized offer",
                "Assign to retention specialist",
                "Offer loyalty program enrollment"
            ])
        
        if customer_features['days_since_last_order'] > 30:
            recommendations.append("Send win-back campaign")
        
        if customer_features['support_tickets'] > 2:
            recommendations.append("Proactive customer support outreach")
        
        if customer_features['email_open_rate'] < 0.2:
            recommendations.append("Review email communication strategy")
        
        if customer_features['return_rate'] > 0.3:
            recommendations.append("Investigate product quality issues")
        
        return recommendations
    
    def _get_key_risk_factors(self, customer_features: pd.Series) -> List[Dict]:
        """Get key risk factors contributing to churn"""
        
        if self.model is None:
            return []
        
        try:
            # Get feature importance
            importance_dict = dict(zip(customer_features.index, self.model.feature_importances_))
            
            # Get top risk factors
            top_factors = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            
            factors = []
            for factor, importance in top_factors:
                # Determine if this is a risk factor
                risk_threshold = self._get_risk_threshold(factor)
                customer_value = float(customer_features[factor])
                
                if self._is_risk_factor(factor, customer_value, risk_threshold):
                    factors.append({
                        'factor': factor,
                        'importance': float(importance),
                        'customer_value': customer_value,
                        'risk_threshold': risk_threshold,
                        'severity': 'high' if customer_value > risk_threshold * 1.5 else 'medium'
                    })
            
            return factors
            
        except Exception as e:
            logging.error(f"Failed to get key risk factors: {e}")
            return []
    
    def _get_risk_threshold(self, factor: str) -> float:
        """Get risk threshold for a specific factor"""
        
        risk_thresholds = {
            'days_since_last_order': 60,
            'days_since_last_login': 30,
            'support_tickets': 3,
            'return_rate': 0.2,
            'failed_payments': 2,
            'app_usage_decline': 0.5
        }
        
        return risk_thresholds.get(factor, 0.5)
    
    def _is_risk_factor(self, factor: str, value: float, threshold: float) -> bool:
        """Determine if a factor value indicates risk"""
        
        # Factors where higher values indicate more risk
        higher_risk_factors = [
            'days_since_last_order', 'days_since_last_login', 'support_tickets',
            'return_rate', 'failed_payments', 'app_usage_decline'
        ]
        
        # Factors where lower values indicate more risk
        lower_risk_factors = [
            'email_open_rate', 'social_engagement', 'orders_per_month'
        ]
        
        if factor in higher_risk_factors:
            return value > threshold
        elif factor in lower_risk_factors:
            return value < threshold
        
        return False

class AdvancedBundlingEngine:
    """
    Advanced product bundling using ML-based association rules and demand correlation
    """
    
    def __init__(self):
        self.association_rules = {}
        self.demand_correlations = {}
        
    def analyze_purchase_patterns(self, transaction_data: pd.DataFrame):
        """Analyze purchase patterns for bundling opportunities"""
        
        try:
            # Create market basket analysis
            basket = transaction_data.groupby(['user_id', 'order_id'])['product_id'].apply(list).reset_index()
            
            # Calculate product co-occurrence
            from itertools import combinations
            co_occurrence = {}
            
            for _, row in basket.iterrows():
                products = row['product_id']
                if len(products) > 1:
                    for combo in combinations(products, 2):
                        pair = tuple(sorted(combo))
                        co_occurrence[pair] = co_occurrence.get(pair, 0) + 1
            
            # Convert to association rules
            total_transactions = len(basket)
            self.association_rules = {}
            
            for pair, count in co_occurrence.items():
                if count >= 5:  # Minimum support threshold
                    support = count / total_transactions
                    
                    # Calculate confidence (A -> B)
                    product_a_count = transaction_data[transaction_data['product_id'] == pair[0]]['order_id'].nunique()
                    product_b_count = transaction_data[transaction_data['product_id'] == pair[1]]['order_id'].nunique()
                    
                    confidence_a_to_b = count / product_a_count if product_a_count > 0 else 0
                    confidence_b_to_a = count / product_b_count if product_b_count > 0 else 0
                    
                    self.association_rules[pair] = {
                        'support': support,
                        'confidence_a_to_b': confidence_a_to_b,
                        'confidence_b_to_a': confidence_b_to_a,
                        'lift': support / ((product_a_count/total_transactions) * (product_b_count/total_transactions)) if product_a_count > 0 and product_b_count > 0 else 0,
                        'transaction_count': count
                    }
            
            logging.info(f"Analyzed {len(self.association_rules)} product associations")
            
        except Exception as e:
            logging.error(f"Purchase pattern analysis failed: {e}")
    
    def get_bundle_recommendations(self, anchor_product: str, top_k: int = 5) -> List[Dict]:
        """Get bundle recommendations for an anchor product"""
        
        bundle_candidates = []
        
        for pair, metrics in self.association_rules.items():
            if anchor_product in pair:
                other_product = pair[1] if pair[0] == anchor_product else pair[0]
                
                # Calculate bundle score
                confidence = metrics['confidence_a_to_b'] if pair[0] == anchor_product else metrics['confidence_b_to_a']
                bundle_score = confidence * metrics['lift'] * metrics['support']
                
                bundle_candidates.append({
                    'product_id': other_product,
                    'bundle_score': bundle_score,
                    'confidence': confidence,
                    'lift': metrics['lift'],
                    'support': metrics['support'],
                    'estimated_uplift': min(confidence * 0.15, 0.25)  # Conservative estimate
                })
        
        # Sort by bundle score and return top_k
        bundle_candidates.sort(key=lambda x: x['bundle_score'], reverse=True)
        return bundle_candidates[:top_k]
    
    def create_smart_bundles(self, product_catalog: pd.DataFrame, max_bundles: int = 20) -> List[Dict]:
        """Create smart product bundles based on analysis"""
        
        smart_bundles = []
        used_products = set()
        
        # Sort association rules by lift * support
        sorted_rules = sorted(
            self.association_rules.items(),
            key=lambda x: x[1]['lift'] * x[1]['support'],
            reverse=True
        )
        
        for pair, metrics in sorted_rules[:max_bundles]:
            if pair[0] not in used_products and pair[1] not in used_products:
                
                # Get product information
                product_a_info = product_catalog[product_catalog['product_id'] == pair[0]]
                product_b_info = product_catalog[product_catalog['product_id'] == pair[1]]
                
                if not product_a_info.empty and not product_b_info.empty:
                    
                    # Calculate bundle pricing
                    price_a = product_a_info.iloc[0]['price']
                    price_b = product_b_info.iloc[0]['price']
                    individual_total = price_a + price_b
                    bundle_discount = 0.1 + (metrics['lift'] - 1) * 0.05  # Dynamic discount based on lift
                    bundle_price = individual_total * (1 - min(bundle_discount, 0.25))
                    
                    smart_bundles.append({
                        'bundle_id': f"bundle_{pair[0]}_{pair[1]}",
                        'product_ids': list(pair),
                        'product_names': [product_a_info.iloc[0]['name'], product_b_info.iloc[0]['name']],
                        'individual_prices': [price_a, price_b],
                        'individual_total': individual_total,
                        'bundle_price': bundle_price,
                        'savings': individual_total - bundle_price,
                        'discount_percentage': bundle_discount,
                        'confidence': max(metrics['confidence_a_to_b'], metrics['confidence_b_to_a']),
                        'lift': metrics['lift'],
                        'support': metrics['support'],
                        'expected_conversion_uplift': min(metrics['lift'] * 0.1, 0.3)
                    })
                    
                    used_products.update(pair)
        
        return smart_bundles

# Merchant Dashboard Integration
class MerchantDashboardService:
    """
    Advanced analytics and insights for merchant dashboard
    """
    
    def __init__(self):
        self.clv_predictor = CustomerLifetimeValuePredictor()
        self.churn_predictor = ChurnPredictor()
        self.bundling_engine = AdvancedBundlingEngine()
    
    def get_merchant_insights(self, merchant_id: str, customer_data: pd.DataFrame,
                            transaction_data: pd.DataFrame, product_data: pd.DataFrame) -> Dict:
        """Get comprehensive merchant insights"""
        
        insights = {
            'merchant_id': merchant_id,
            'generated_at': datetime.now().isoformat(),
            'customer_segments': {},
            'churn_analysis': {},
            'bundling_opportunities': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
        try:
            # Customer Segmentation by CLV
            if not customer_data.empty:
                # Add dummy CLV for demonstration
                customer_data['actual_clv'] = np.random.lognormal(6, 1, len(customer_data))
                
                # Train CLV model
                self.clv_predictor.train_clv_model(customer_data, 'actual_clv')
                
                # Segment customers
                high_value = len(self.clv_predictor.clv_segments.get('high_value', []))
                medium_value = len(self.clv_predictor.clv_segments.get('medium_value', []))
                low_value = len(self.clv_predictor.clv_segments.get('low_value', []))
                
                insights['customer_segments'] = {
                    'total_customers': len(customer_data),
                    'high_value_customers': high_value,
                    'medium_value_customers': medium_value,
                    'low_value_customers': low_value,
                    'high_value_percentage': (high_value / len(customer_data)) * 100
                }
            
            # Churn Analysis
            if not customer_data.empty:
                # Add dummy churn labels for demonstration
                customer_data['is_churned'] = np.random.choice([0, 1], len(customer_data), p=[0.8, 0.2])
                
                # Train churn model
                self.churn_predictor.train_churn_model(customer_data, 'is_churned')
                
                # Calculate churn metrics
                at_risk_customers = customer_data[customer_data['is_churned'] == 1]
                
                insights['churn_analysis'] = {
                    'total_at_risk': len(at_risk_customers),
                    'churn_rate': (len(at_risk_customers) / len(customer_data)) * 100,
                    'estimated_revenue_at_risk': float(at_risk_customers['total_spent'].sum()),
                    'retention_priority_list': at_risk_customers.head(10)['customer_id'].tolist()
                }
            
            # Bundling Opportunities
            if not transaction_data.empty:
                self.bundling_engine.analyze_purchase_patterns(transaction_data)
                smart_bundles = self.bundling_engine.create_smart_bundles(product_data, max_bundles=5)
                
                insights['bundling_opportunities'] = {
                    'total_bundle_opportunities': len(smart_bundles),
                    'top_bundles': smart_bundles[:3],
                    'estimated_revenue_uplift': sum(bundle['expected_conversion_uplift'] for bundle in smart_bundles) * 1000  # Placeholder calculation
                }
            
            # Performance Metrics
            insights['performance_metrics'] = self._calculate_performance_metrics(
                customer_data, transaction_data, product_data
            )
            
            # Strategic Recommendations
            insights['recommendations'] = self._generate_strategic_recommendations(insights)
            
        except Exception as e:
            logging.error(f"Merchant insights generation failed: {e}")
            insights['error'] = str(e)
        
        return insights
    
    def _calculate_performance_metrics(self, customer_data: pd.DataFrame,
                                     transaction_data: pd.DataFrame,
                                     product_data: pd.DataFrame) -> Dict:
        """Calculate key performance metrics"""
        
        metrics = {}
        
        try:
            if not customer_data.empty:
                metrics['average_clv'] = customer_data['total_spent'].mean()
                metrics['customer_acquisition_trend'] = 'stable'  # Would calculate from time series
                
            if not transaction_data.empty:
                metrics['average_order_value'] = transaction_data.groupby('order_id')['price'].sum().mean()
                metrics['repeat_purchase_rate'] = (transaction_data['user_id'].value_counts() > 1).mean()
                
            if not product_data.empty:
                metrics['product_performance'] = {
                    'total_products': len(product_data),
                    'out_of_stock_rate': 0.05,  # Placeholder
                    'top_performing_category': 'electronics'  # Placeholder
                }
                
        except Exception as e:
            logging.error(f"Performance metrics calculation failed: {e}")
        
        return metrics
    
    def _generate_strategic_recommendations(self, insights: Dict) -> List[str]:
        """Generate strategic recommendations based on insights"""
        
        recommendations = []
        
        try:
            # CLV-based recommendations
            if insights['customer_segments'].get('high_value_percentage', 0) < 20:
                recommendations.append("Focus on increasing customer lifetime value through loyalty programs")
            
            # Churn-based recommendations
            if insights['churn_analysis'].get('churn_rate', 0) > 15:
                recommendations.append("Implement proactive retention campaigns for at-risk customers")
            
            # Bundling recommendations
            if insights['bundling_opportunities'].get('total_bundle_opportunities', 0) > 0:
                recommendations.append("Launch product bundling campaigns to increase average order value")
            
            # General recommendations
            recommendations.extend([
                "Implement personalized recommendation system",
                "Develop customer segmentation-based marketing campaigns",
                "Monitor and optimize key performance indicators regularly"
            ])
            
        except Exception as e:
            logging.error(f"Strategic recommendations generation failed: {e}")
        
        return recommendations

# Factory functions
def create_clv_predictor() -> CustomerLifetimeValuePredictor:
    """Create and return a CLV predictor instance"""
    return CustomerLifetimeValuePredictor()

def create_churn_predictor() -> ChurnPredictor:
    """Create and return a churn predictor instance"""
    return ChurnPredictor()

def create_bundling_engine() -> AdvancedBundlingEngine:
    """Create and return a bundling engine instance"""
    return AdvancedBundlingEngine()

def create_merchant_dashboard_service() -> MerchantDashboardService:
    """Create and return a merchant dashboard service instance"""
    return MerchantDashboardService()
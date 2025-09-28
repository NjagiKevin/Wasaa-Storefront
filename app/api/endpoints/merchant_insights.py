"""
Enhanced API endpoints for merchant insights and advanced ML capabilities
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import logging

from app.db import crud
from app.api.dependencies import get_db
from app.ml_models.business_intelligence import create_merchant_dashboard_service
from app.ml_models.advanced_recommender import create_advanced_recommender
from app.ml_models.advanced_forecasting import create_advanced_forecaster
from app.ml_models.advanced_fraud_detection import create_advanced_fraud_detector
from app.utils.cache import cache_get, cache_set
from app.utils.metrics import recommendation_requests_total, recommendation_request_latency_seconds
import time

router = APIRouter(prefix="/merchant", tags=["Merchant Insights"])

# Initialize ML services
dashboard_service = create_merchant_dashboard_service()
recommender_service = create_advanced_recommender()
forecaster_service = create_advanced_forecaster()
fraud_detector = create_advanced_fraud_detector()

@router.get("/{merchant_id}/insights")
async def get_merchant_insights(
    merchant_id: str,
    include_forecasting: bool = True,
    include_churn_analysis: bool = True,
    include_bundling: bool = True,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive merchant insights with advanced ML analytics
    """
    try:
        # Check cache first
        cache_key = f"merchant_insights:{merchant_id}"
        cached_insights = cache_get(cache_key)
        if cached_insights:
            return cached_insights
        
        # Get merchant data
        customer_data = crud.get_merchant_customers(db, merchant_id)
        transaction_data = crud.get_merchant_transactions(db, merchant_id)
        product_data = crud.get_merchant_products(db, merchant_id)
        
        # Convert to DataFrames
        customer_df = pd.DataFrame([{
            'customer_id': c.customer_id,
            'registration_date': c.created_at,
            'last_order_date': c.last_order_date,
            'total_orders': c.total_orders,
            'total_spent': c.total_spent
        } for c in customer_data])
        
        transaction_df = pd.DataFrame([{
            'user_id': t.user_id,
            'order_id': t.order_id,
            'product_id': t.product_id,
            'price': t.amount
        } for t in transaction_data])
        
        product_df = pd.DataFrame([{
            'product_id': p.product_id,
            'name': p.name,
            'price': p.price,
            'category': p.category
        } for p in product_data])
        
        # Generate comprehensive insights
        insights = dashboard_service.get_merchant_insights(
            merchant_id, customer_df, transaction_df, product_df
        )
        
        # Add forecasting insights if requested
        if include_forecasting and not product_df.empty:
            insights['demand_forecasting'] = await get_demand_forecasting_insights(
                merchant_id, product_df, db
            )
        
        # Cache results for 1 hour
        cache_set(cache_key, insights, ttl_seconds=3600)
        
        return insights
        
    except Exception as e:
        logging.error(f"Merchant insights failed for {merchant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{merchant_id}/clv-predictions")
async def get_customer_clv_predictions(
    merchant_id: str,
    top_k: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get Customer Lifetime Value predictions for merchant's customers
    """
    try:
        customers = crud.get_merchant_customers(db, merchant_id)
        
        clv_predictions = []
        for customer in customers[:top_k]:
            customer_data = {
                'customer_id': customer.customer_id,
                'registration_date': customer.created_at,
                'last_order_date': customer.last_order_date,
                'total_orders': customer.total_orders,
                'total_spent': customer.total_spent
            }
            
            clv_result = dashboard_service.clv_predictor.predict_clv(customer_data)
            clv_result['customer_id'] = customer.customer_id
            clv_predictions.append(clv_result)
        
        return {
            'merchant_id': merchant_id,
            'total_predictions': len(clv_predictions),
            'predictions': clv_predictions,
            'segments': {
                'high_value': len([p for p in clv_predictions if p['segment'] == 'high_value']),
                'medium_value': len([p for p in clv_predictions if p['segment'] == 'medium_value']),
                'low_value': len([p for p in clv_predictions if p['segment'] == 'low_value'])
            }
        }
        
    except Exception as e:
        logging.error(f"CLV predictions failed for {merchant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{merchant_id}/churn-risk")
async def get_churn_risk_analysis(
    merchant_id: str,
    risk_level_filter: Optional[str] = None,
    top_k: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get churn risk analysis for merchant's customers
    """
    try:
        customers = crud.get_merchant_customers(db, merchant_id)
        
        churn_predictions = []
        for customer in customers[:top_k]:
            customer_data = {
                'customer_id': customer.customer_id,
                'registration_date': customer.created_at,
                'last_order_date': customer.last_order_date,
                'total_orders': customer.total_orders,
                'total_spent': customer.total_spent
            }
            
            churn_result = dashboard_service.churn_predictor.predict_churn_probability(customer_data)
            churn_result['customer_id'] = customer.customer_id
            
            # Filter by risk level if specified
            if risk_level_filter is None or churn_result['risk_level'] == risk_level_filter:
                churn_predictions.append(churn_result)
        
        # Sort by churn probability (highest risk first)
        churn_predictions.sort(key=lambda x: x['churn_probability'], reverse=True)
        
        return {
            'merchant_id': merchant_id,
            'total_customers_analyzed': len(customers),
            'at_risk_customers': len(churn_predictions),
            'risk_distribution': {
                'critical': len([p for p in churn_predictions if p['risk_level'] == 'critical']),
                'high': len([p for p in churn_predictions if p['risk_level'] == 'high']),
                'medium': len([p for p in churn_predictions if p['risk_level'] == 'medium']),
                'low': len([p for p in churn_predictions if p['risk_level'] == 'low'])
            },
            'predictions': churn_predictions[:50]  # Return top 50 at-risk customers
        }
        
    except Exception as e:
        logging.error(f"Churn risk analysis failed for {merchant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{merchant_id}/bundling-opportunities")
async def get_bundling_opportunities(
    merchant_id: str,
    anchor_product: Optional[str] = None,
    max_bundles: int = 20,
    db: Session = Depends(get_db)
):
    """
    Get product bundling opportunities for merchant
    """
    try:
        # Get transaction data
        transactions = crud.get_merchant_transactions(db, merchant_id)
        products = crud.get_merchant_products(db, merchant_id)
        
        transaction_df = pd.DataFrame([{
            'user_id': t.user_id,
            'order_id': t.order_id,
            'product_id': t.product_id,
            'price': t.amount
        } for t in transactions])
        
        product_df = pd.DataFrame([{
            'product_id': p.product_id,
            'name': p.name,
            'price': p.price,
            'category': p.category
        } for p in products])
        
        # Analyze purchase patterns
        dashboard_service.bundling_engine.analyze_purchase_patterns(transaction_df)
        
        if anchor_product:
            # Get recommendations for specific product
            bundle_recommendations = dashboard_service.bundling_engine.get_bundle_recommendations(
                anchor_product, top_k=10
            )
            return {
                'merchant_id': merchant_id,
                'anchor_product': anchor_product,
                'bundle_recommendations': bundle_recommendations
            }
        else:
            # Get smart bundles
            smart_bundles = dashboard_service.bundling_engine.create_smart_bundles(
                product_df, max_bundles=max_bundles
            )
            return {
                'merchant_id': merchant_id,
                'total_bundle_opportunities': len(smart_bundles),
                'smart_bundles': smart_bundles,
                'estimated_revenue_impact': sum(bundle['expected_conversion_uplift'] for bundle in smart_bundles) * 1000
            }
        
    except Exception as e:
        logging.error(f"Bundling opportunities analysis failed for {merchant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{merchant_id}/demand-forecast")
async def get_demand_forecast(
    merchant_id: str,
    product_id: Optional[str] = None,
    forecast_days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get demand forecasting for merchant's products
    """
    try:
        # Get historical sales data
        sales_data = crud.get_merchant_sales_history(db, merchant_id, product_id)
        
        if not sales_data:
            raise HTTPException(status_code=404, detail="No sales data found")
        
        # Convert to DataFrame
        sales_df = pd.DataFrame([{
            'date': s.date,
            'product_id': s.product_id,
            'demand': s.quantity_sold
        } for s in sales_data])
        
        forecasts = {}
        
        if product_id:
            # Forecast for specific product
            product_data = sales_df[sales_df['product_id'] == product_id]
            if not product_data.empty:
                # Train models if not already trained
                forecaster_service.train_all_models(product_data, product_id)
                
                # Generate ensemble forecast
                forecast_result = forecaster_service.ensemble_forecast(
                    product_id, periods=forecast_days, last_data=product_data
                )
                forecasts[product_id] = forecast_result
        else:
            # Forecast for all products
            for pid in sales_df['product_id'].unique():
                product_data = sales_df[sales_df['product_id'] == pid]
                if len(product_data) > 30:  # Minimum data requirement
                    try:
                        forecaster_service.train_all_models(product_data, pid)
                        forecast_result = forecaster_service.ensemble_forecast(
                            pid, periods=forecast_days, last_data=product_data
                        )
                        forecasts[pid] = forecast_result
                    except Exception as e:
                        logging.warning(f"Forecast failed for product {pid}: {e}")
        
        return {
            'merchant_id': merchant_id,
            'forecast_period_days': forecast_days,
            'products_forecasted': len(forecasts),
            'forecasts': forecasts
        }
        
    except Exception as e:
        logging.error(f"Demand forecasting failed for {merchant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{merchant_id}/context-recommendations")
async def get_context_aware_recommendations(
    user_id: str,
    merchant_id: str,
    user_location: Optional[List[float]] = None,
    top_k: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get context-aware product recommendations
    """
    endpoint = "/merchant/{merchant_id}/context-recommendations"
    recommendation_requests_total.labels(endpoint=endpoint).inc()
    t0 = time.perf_counter()
    
    try:
        # Get merchant's products
        products = crud.get_merchant_products(db, merchant_id)
        product_ids = [str(p.product_id) for p in products]
        
        # Convert location to tuple if provided
        location_tuple = tuple(user_location) if user_location else None
        
        # Get context-aware recommendations
        recommendations = recommender_service.get_context_aware_recommendations(
            user_id=user_id,
            top_k=top_k,
            user_location=location_tuple,
            all_products=product_ids
        )
        
        recommendation_request_latency_seconds.labels(endpoint=endpoint).observe(time.perf_counter() - t0)
        
        return {
            'user_id': user_id,
            'merchant_id': merchant_id,
            'user_location': user_location,
            'recommendations': recommendations,
            'context_features_used': [
                'temporal_features',
                'location_features' if user_location else None,
                'trending_analysis',
                'collaborative_filtering',
                'content_based_filtering'
            ]
        }
        
    except Exception as e:
        logging.error(f"Context-aware recommendations failed: {e}")
        recommendation_request_latency_seconds.labels(endpoint=endpoint).observe(time.perf_counter() - t0)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{merchant_id}/fraud-check")
async def check_transaction_fraud(
    merchant_id: str,
    transaction_data: Dict,
    db: Session = Depends(get_db)
):
    """
    Real-time fraud detection for transactions
    """
    try:
        # Add merchant context to transaction data
        transaction_data['merchant_id'] = merchant_id
        
        # Get fraud prediction
        fraud_result = fraud_detector.predict_fraud_probability(transaction_data)
        
        return {
            'merchant_id': merchant_id,
            'transaction_id': transaction_data.get('transaction_id', 'unknown'),
            'fraud_analysis': fraud_result,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Fraud detection failed for merchant {merchant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{merchant_id}/stock-alerts")
async def get_stock_out_predictions(
    merchant_id: str,
    alert_threshold_days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Get real-time stock-out predictions and alerts
    """
    try:
        # Get inventory data
        inventory_data = crud.get_merchant_inventory(db, merchant_id)
        
        stock_alerts = []
        
        for item in inventory_data:
            if item.current_stock <= item.reorder_level:
                # Get demand forecast for this product
                try:
                    # Simplified forecast calculation
                    daily_demand = item.weekly_sales / 7 if item.weekly_sales else 1
                    days_until_stockout = item.current_stock / daily_demand if daily_demand > 0 else 999
                    
                    if days_until_stockout <= alert_threshold_days:
                        stock_alerts.append({
                            'product_id': str(item.product_id),
                            'product_name': item.product_name,
                            'current_stock': item.current_stock,
                            'reorder_level': item.reorder_level,
                            'daily_demand_estimate': daily_demand,
                            'days_until_stockout': round(days_until_stockout, 1),
                            'urgency_level': 'critical' if days_until_stockout <= 3 else 'high' if days_until_stockout <= 5 else 'medium',
                            'recommended_reorder_quantity': max(item.reorder_quantity, daily_demand * 14)  # 2 weeks supply
                        })
                except Exception as e:
                    logging.warning(f"Stock prediction failed for {item.product_id}: {e}")
        
        # Sort by urgency
        stock_alerts.sort(key=lambda x: x['days_until_stockout'])
        
        return {
            'merchant_id': merchant_id,
            'alert_threshold_days': alert_threshold_days,
            'total_alerts': len(stock_alerts),
            'critical_alerts': len([a for a in stock_alerts if a['urgency_level'] == 'critical']),
            'high_priority_alerts': len([a for a in stock_alerts if a['urgency_level'] == 'high']),
            'stock_alerts': stock_alerts
        }
        
    except Exception as e:
        logging.error(f"Stock alerts failed for merchant {merchant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{merchant_id}/train-models")
async def train_merchant_models(
    merchant_id: str,
    background_tasks: BackgroundTasks,
    model_types: List[str] = ["recommendations", "forecasting", "fraud"],
    db: Session = Depends(get_db)
):
    """
    Trigger training of ML models for merchant (background task)
    """
    try:
        def train_models_task():
            """Background task to train models"""
            try:
                logging.info(f"Starting model training for merchant {merchant_id}")
                
                if "recommendations" in model_types:
                    # Train recommendation models
                    customer_interactions = crud.get_merchant_customer_interactions(db, merchant_id)
                    if customer_interactions:
                        interaction_df = pd.DataFrame([{
                            'user_id': i.user_id,
                            'product_id': i.product_id,
                            'rating': i.interaction_score,
                            'timestamp': i.created_at
                        } for i in customer_interactions])
                        
                        recommender_service.train_matrix_factorization(
                            interaction_df.pivot_table(
                                index='user_id', columns='product_id', values='rating', fill_value=0
                            ).values
                        )
                        logging.info(f"Recommendation models trained for {merchant_id}")
                
                if "forecasting" in model_types:
                    # Train forecasting models
                    sales_data = crud.get_merchant_sales_history(db, merchant_id)
                    if sales_data:
                        sales_df = pd.DataFrame([{
                            'date': s.date,
                            'product_id': s.product_id,
                            'demand': s.quantity_sold
                        } for s in sales_data])
                        
                        for product_id in sales_df['product_id'].unique():
                            product_data = sales_df[sales_df['product_id'] == product_id]
                            if len(product_data) > 30:
                                forecaster_service.train_all_models(product_data, product_id)
                        
                        logging.info(f"Forecasting models trained for {merchant_id}")
                
                if "fraud" in model_types:
                    # Train fraud detection models
                    transaction_data = crud.get_merchant_fraud_training_data(db, merchant_id)
                    if transaction_data and len(transaction_data) > 100:
                        fraud_df = pd.DataFrame([{
                            'user_id': t.user_id,
                            'amount': t.amount,
                            'merchant_id': merchant_id,
                            'is_fraud': t.is_fraud
                        } for t in transaction_data])
                        
                        fraud_detector.train_all_models(fraud_df)
                        logging.info(f"Fraud detection models trained for {merchant_id}")
                
                logging.info(f"Model training completed for merchant {merchant_id}")
                
            except Exception as e:
                logging.error(f"Model training failed for merchant {merchant_id}: {e}")
        
        # Add training task to background queue
        background_tasks.add_task(train_models_task)
        
        return {
            'merchant_id': merchant_id,
            'status': 'training_initiated',
            'model_types': model_types,
            'message': 'Model training started in background. Check logs for progress.'
        }
        
    except Exception as e:
        logging.error(f"Failed to initiate model training for {merchant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper function for demand forecasting insights
async def get_demand_forecasting_insights(merchant_id: str, product_df: pd.DataFrame, db: Session) -> Dict:
    """Get demand forecasting insights"""
    try:
        insights = {
            'total_products': len(product_df),
            'forecasts_available': 0,
            'high_demand_products': [],
            'declining_demand_products': [],
            'forecast_accuracy': {}
        }
        
        # This would be implemented with actual historical data and forecasting
        # For now, return placeholder insights
        insights['forecasts_available'] = min(len(product_df), 10)
        insights['forecast_accuracy'] = {
            'prophet': 0.85,
            'arima': 0.82,
            'lstm': 0.88,
            'ensemble': 0.90
        }
        
        return insights
        
    except Exception as e:
        logging.error(f"Demand forecasting insights failed: {e}")
        return {}
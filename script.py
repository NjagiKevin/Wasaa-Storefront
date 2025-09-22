import os

# Base project folder
base_dir = "wasaa_storefront"

# Updated folder structure
structure = {
    "app": {
        "api": {"endpoints": ["recommendations.py", "merchants.py", "orders.py", "health.py"],
                "dependencies.py": ""},
        "core": ["config.py", "logging.py", "security.py"],
        "models": [
            "user.py", "merchant.py", "store.py", "product.py", "inventory.py",
            "order.py", "payment.py", "delivery.py", "review.py", "analytics.py", "__init__.py"
        ],
        "services": ["recommendation_service.py", "demand_forecast_service.py", "pricing_service.py"],
        "ml_models": ["collaborative_filtering.pkl", "demand_forecast_lstm.pkl", "pricing_model.pkl"],
        "db": {"crud": [
                    "user_crud.py", "merchant_crud.py", "store_crud.py", "product_crud.py",
                    "inventory_crud.py", "order_crud.py", "payment_crud.py", "delivery_crud.py",
                    "review_crud.py", "analytics_crud.py"
               ],
               "base.py": "", "session.py": ""},
        "utils": ["preprocessing.py", "kafka_producer.py", "cache.py"]
    },
    "airflow": {"dags": ["etl_transactions.py", "etl_ads.py", "ml_pipeline.py"],
                "plugins": [],
                "docker-compose.yaml": ""},
    "tests": {"unit": [
                    "test_recommendations.py", "test_forecasting.py", "test_pricing.py",
                    "test_merchant_model.py", "test_store_model.py", "test_product_model.py"
              ],
              "integration": ["test_api_endpoints.py", "test_db_integration.py"]},
    "scripts": ["train_models.py", "update_features.py", "init_db.py"],
    "files": ["requirements.txt", "pyproject.toml", "setup.cfg", "README.md", ".env", "docker-compose.yml", "Dockerfile"]
}

def create_structure(base_path, struct):
    for key, value in struct.items():
        path = os.path.join(base_path, key)
        if isinstance(value, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, value)
        elif isinstance(value, list):
            os.makedirs(path, exist_ok=True)
            for f in value:
                file_path = os.path.join(path, f)
                with open(file_path, "w") as fp:
                    fp.write(f"# {f} placeholder\n")
        else:
            # Single file
            file_path = os.path.join(base_path, key)
            with open(file_path, "w") as fp:
                fp.write(f"# {key} placeholder\n")

# Create base directory
os.makedirs(base_dir, exist_ok=True)

# Create structure
create_structure(base_dir, structure)

print(f"Project structure created under '{base_dir}'")

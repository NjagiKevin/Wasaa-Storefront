# Wasaa Storefront ML Service

AI-powered services for the WasaaChat Storefront, providing recommendations, demand forecasting, fraud/risk scoring, and operational observability. The stack combines a FastAPI application, a BentoML service for model serving, MLflow for experiment tracking, Redis for caching, and PostgreSQL for persistence. Airflow is included for orchestration and scheduled jobs.


## Highlights
- FastAPI web API with modular endpoints (health, recommendations, forecasting, fraud, metrics)
- Hybrid recommendation logic with fallbacks (context-aware, collaborative/content-based heuristics)
- Demand forecasting service with simple heuristics and hooks for advanced models
- Fraud scoring with rule-based stub and advanced/fallback pathways
- BentoML packaging for ML microservice deployment
- MLflow tracking integration and Postgres-backed stores
- Redis cache for performance and Prometheus metrics endpoint for observability
- Docker and docker-compose for local, reproducible environments


## Architecture
- API (FastAPI): app/main, app/api/endpoints/*
  - Endpoints for recommendations, forecasting, fraud scoring, metrics, and health
  - SQLAlchemy models and sessions; tables auto-created on startup
  - Prometheus metrics at /metrics
- ML Service (BentoML): storefront_ml_service.py packaged by bentofile.yaml
  - APIs: recommend, forecast, fraud_check (with robust fallbacks)
- Data + Infra (docker-compose):
  - Postgres instances for Airflow, MLflow, BentoML
  - Redis cache
  - MLflow server with artifact storage
  - Optional Airflow webserver/scheduler/init services
- Tracking & Ops: MLflow (http://localhost:5000), Prometheus scrape at /metrics, logs in ./logs


## Tech Stack
- Python 3.11
- FastAPI, Uvicorn
- SQLAlchemy, psycopg2
- Redis, Prometheus client
- ML: numpy, pandas, scikit-learn, xgboost, lightgbm, statsmodels (CPU-only)
- BentoML (service packaging)
- MLflow (tracking)
- Docker & docker-compose


## Directory Structure (key paths)
- app/
  - api/endpoints: health.py, recommendations.py, forecasting.py, fraud.py, metrics.py
  - core: config.py, logging.py
  - db: base.py, session.py, models.py, crud/
  - services: recommendation_service.py, demand_forecast_service.py, fraud_service.py, pricing_service.py
  - utils: metrics.py, cache.py
- storefront_ml_service.py (BentoML service)
- bentofile.yaml (Bento build spec)
- requirements*.txt
- Dockerfile, docker-compose.yaml
- docs/: ML-Endpoints.md, Postman collection/envs
- airflow/: Dockerfiles and DAGs (if any)
- logs/: rotating logs (created at runtime)


## Setup

### Prerequisites
- Python 3.11
- Docker and Docker Compose
- Make sure port mappings are free (8000, 5000, 3000, 3001, 8081, etc.)

### Environment Variables
The app reads settings from environment variables and/or a .env file. At minimum:

- API/App
  - APP_NAME: Application name
  - DEBUG: true/false
  - SECRET_KEY: JWT/crypto secret (set a strong value)
  - LOG_LEVEL: e.g. INFO, DEBUG
- Database
  - DATABASE_URL: SQLAlchemy URL for the API (e.g., postgresql+psycopg2://USER:PASS@HOST:PORT/DB)
  - Or component vars used in docker-compose for the API container: DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME
- Cache
  - REDIS_URL (or REDIS_HOST/REDIS_PORT for docker-compose)
- Tracking/ML
  - MLFLOW_TRACKING_URI (e.g., http://mlflow:5000 when using docker compose)
  - BENTOML_HOME, BENTOML_MODEL_STORE (if applicable)

Do not commit real secrets. Use placeholders or a local .env.


## Run Locally (no Docker)
1) Create and activate a virtual environment

- Windows (PowerShell)
  - python -m venv .venv
  - .\\.venv\\Scripts\\Activate.ps1
- macOS/Linux
  - python3 -m venv .venv
  - source .venv/bin/activate

2) Install dependencies
- pip install -r requirements.txt

3) Configure environment
- Set DATABASE_URL, REDIS_URL, SECRET_KEY, etc. (you can create a .env in the repo root)

4) Start the API
- uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

5) Open
- Docs (Swagger): http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/health
- Metrics (Prometheus): http://localhost:8000/metrics

Tables are auto-created on startup via SQLAlchemy Base.metadata.create_all.


## Run with Docker Compose
This repository includes a docker-compose.yaml that provisions:
- Postgres for Airflow, MLflow, and BentoML
- Redis
- MLflow server
- BentoML service
- FastAPI app (api)
- Airflow webserver, scheduler, and init job

Steps:
1) Create the external network required by the compose file (one-time):
- docker network create storefront-network

2) (Optional) Prepare a .env file for the api service environment (DB_* variables, SECRET_KEY, etc.)

3) Start services:
- docker compose up -d --build

4) Access:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/v1/health
- Metrics: http://localhost:8000/metrics
- MLflow UI: http://localhost:5000
- Airflow Webserver: http://localhost:8081
- BentoML Service: http://localhost:3000 (health at /healthz or as exposed by your Dockerfile.bentoml)

5) Stop services:
- docker compose down


## BentoML Service
The Bento service is defined in storefront_ml_service.py and packaged via bentofile.yaml.

- Build a Bento:
  - pip install bentoml (if not already installed)
  - bentoml build -f bentofile.yaml

- Serve locally (from your local Bento store):
  - bentoml serve wasaa-storefront-ml:2.0.0
  - Default ports are typically 3000 (API) and 3001 (UI) if configured; see docker-compose and bentofile

- Test requests (examples): see docs/ML-Endpoints.md and test_requests.json


## API Endpoints (FastAPI)
- GET /api/v1/health – health check (DB connectivity and optional downstream checks)
- GET /api/v1/recommendations/products/{user_id} – recommended products for a user (uses Redis cache if configured)
- POST /api/v1/forecasting/demand – demand scores for a list of products
- POST /api/v1/fraud/score – fraud risk score for a transaction
- GET /metrics – Prometheus metrics
- Root / – welcome payload with links

Full request/response examples are in docs/ML-Endpoints.md and the Postman collection in docs/.


## Development Notes
- Logging: configured via app/core/logging.py with rotating file handler. Logs go to ./logs/app.log
- CORS: currently permissive (allow_origins=['*']) – tighten for production
- Auth: JWT helpers and optional auth dependency are present. Endpoints above do not strictly require auth by default; adjust per needs
- DB: SQLAlchemy engine configured in app/db/session.py; change DATABASE_URL as needed
- Caching: app/utils/cache.py is a placeholder; integrate Redis client for cache_get/cache_set in production
- Metrics: app/utils/metrics.py exposes Prometheus counters/histograms


## Testing
- Run tests (unit/integration):
  - pytest -q
- BentoML service test:
  - python test_bentoml_service.py


## Troubleshooting
- Port conflicts: stop other services or change exposed ports in docker-compose.yaml
- Database connectivity: check DATABASE_URL (local) or DB_* envs (compose). Ensure the external network exists for compose
- MLflow not reachable: confirm http://localhost:5000 is accessible and health checks are green
- Missing advanced models: storefront_ml_service.py includes robust fallbacks; advanced components will be skipped if not available
- Windows paths: When running in Git Bash, prefer forward slashes for paths (e.g., F:/WEBMASTERS/...)


## Deployment
- Containerized deployment recommended (Dockerfile provided). The Dockerfile runs uvicorn app.main:app on port 8000
- For ML service, build and deploy Bento (Dockerfile.bentoml) and point API to the Bento model store if integrated
- Configure environment-specific secrets securely (avoid committing real values)


## References
- docs/ML-Endpoints.md – cURL examples and endpoint details
- docs/ML.postman_collection.json – Postman collection
- BENTOML_DEPLOYMENT_GUIDE.md – BentoML deployment guidance
- DATABASE_CONNECTION_README.md – Database notes
- 100_PERCENT_AI_IMPLEMENTATION_SUMMARY.md – Overview of AI implementation

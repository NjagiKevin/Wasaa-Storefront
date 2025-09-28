# 📡 External Database Connection Setup

This ML system connects to the **main Storefront microservice database** to access business data for machine learning training and inference.

## 🏗️ Architecture Overview

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│                     │    │                     │    │                     │
│   Storefront        │    │  Storefront ML      │    │   ML Components     │
│   Microservice      │────│  System             │────│                     │
│                     │    │                     │    │  • Airflow          │
│  ┌───────────────┐  │    │  ┌───────────────┐  │    │  • MLflow           │
│  │ storefront_   │  │    │  │ Connects to   │  │    │  • Redis            │
│  │ postgres      │◄─┼────┼─►│ External DB   │  │    │  • FastAPI          │
│  │               │  │    │  │               │  │    │                     │
│  │ - users       │  │    │  └───────────────┘  │    └─────────────────────┘
│  │ - products    │  │    │                     │
│  │ - orders      │  │    │  Internal DBs:      │
│  │ - transactions│  │    │  • airflow_db       │
│  └───────────────┘  │    │  • mlflow_db        │
│                     │    │                     │
└─────────────────────┘    └─────────────────────┘
```

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# Backend Database (from Wasaa-Storefront-Backend)
# Use container name and internal port for container-to-container communication
DB_HOST=storefront_postgres
DB_PORT=5432
DB_USER=storefront_user
DB_PASS=storefront_secure_pass
DB_NAME=storefront_db

# Mirror to POSTGRES_* for code that reads these directly
POSTGRES_SERVER=storefront_postgres
POSTGRES_PORT=5432
POSTGRES_USER=storefront_user
POSTGRES_PASSWORD=storefront_secure_pass
POSTGRES_DB=storefront_db
```

### Docker Compose Configuration

The `docker-compose.yaml` passes these environment variables to:

1. **Airflow services** - for accessing business data in DAGs
2. **FastAPI service** - for ML inference and training
3. **MLflow** - uses its own separate database for experiment tracking

## 🌐 Network Requirements

Both the Storefront microservice and ML system must be on the same Docker network:

```bash
# Create the shared network (if not exists)
docker network create storefront-network

# The Storefront microservice should already be on this network
# The ML system joins this network to access storefront_postgres
```

## 🧪 Testing Connection

Use the provided connection test script:

```bash
# Test connection to external database
python test_db_connection.py
```

This will:
- ✅ Test connection to `storefront_postgres`
- 📊 List available tables and row counts  
- 🤖 Check ML-relevant tables (users, products, orders, etc.)
- 📈 Assess ML readiness based on data volume

## 🚀 Deployment Steps

1. **Ensure Storefront microservice is running**:
   ```bash
   # The main storefront service should be running with storefront_postgres
   docker ps | grep storefront_postgres
   ```

2. **Start ML system**:
   ```bash
   # Quick start
   ./setup_ml_stack.sh
   
   # Or manual start
   docker-compose up --build -d
   ```

3. **Verify connection**:
   ```bash
   python test_db_connection.py
   ```

## 📊 Data Access Pattern

### ML Training Data Sources:
- **Users table**: Customer demographics, registration info
- **Products table**: Product catalog, categories, pricing
- **Orders table**: Purchase history, order values, timestamps  
- **Transactions table**: Payment details, success/failure rates
- **User Interactions**: Views, clicks, cart additions
- **Reviews**: Customer feedback, ratings

### Data Flow:
1. **ML DAGs** query external DB for training data
2. **Models** are trained and stored in MLflow
3. **FastAPI** serves predictions using cached models
4. **Real-time inference** connects to external DB for features

## 🔒 Security Considerations

- Database credentials are managed via environment variables
- Connection pooling prevents connection exhaustion
- Read-only access recommended for most ML operations
- Sensitive data should be handled according to privacy policies

## 🐛 Troubleshooting

### Connection Failed?

1. **Check if Storefront microservice is running**:
   ```bash
   docker ps | grep storefront
   ```

2. **Verify network connectivity**:
   ```bash
   docker network ls | grep storefront-network
   ```

3. **Test database credentials**:
   ```bash
   # Update .env with correct credentials from Storefront service
   ```

4. **Check logs**:
   ```bash
   docker-compose logs api
   docker-compose logs airflow-webserver
   ```

### Common Issues:

- **Network not found**: Create `storefront-network` 
- **Permission denied**: Check database user privileges
- **Connection timeout**: Ensure firewall allows connections
- **Tables not found**: Verify Storefront database schema

## 📈 ML Data Requirements

For optimal ML performance, ensure:

| Table | Minimum Rows | Purpose |
|-------|--------------|---------|
| `users` | 100+ | User profiling, segmentation |
| `products` | 50+ | Product recommendations |  
| `orders` | 500+ | Demand forecasting |
| `user_interactions` | 1,000+ | Collaborative filtering |
| `reviews` | 200+ | Sentiment analysis |

## 🎯 Next Steps

After successful connection:
1. ✅ Run ML training DAGs in Airflow
2. ✅ Monitor model performance in MLflow  
3. ✅ Access ML APIs for predictions
4. ✅ Set up automated retraining schedules

---

**📞 Support**: If connection issues persist, verify the main Storefront microservice configuration and ensure database credentials match between both systems.
#!/bin/bash

# Setup script for Wasaa Storefront ML Stack
# Based on adsmanager-ml template

set -e

echo "🚀 Setting up Wasaa Storefront ML Stack..."
echo "📡 This ML system connects to the main Storefront microservice database"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

print_success "Docker is running ✓"

# Check if network exists, create if not
if ! docker network ls | grep -q "storefront-network"; then
    print_status "Creating storefront-network..."
    docker network create storefront-network
    print_success "Network created ✓"
else
    print_success "Network storefront-network already exists ✓"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data models logs scripts airflow/logs airflow/plugins
print_success "Directories created ✓"

# Check if .env file exists
if [ ! -f .env ]; then
    print_error ".env file not found. Please ensure it exists with proper configuration."
    exit 1
fi

print_success ".env file found ✓"

# Build and start services
print_status "Building and starting ML stack..."
docker-compose up --build -d

print_status "Waiting for services to be healthy..."
sleep 30

# Check service health
print_status "Checking service health..."

services=("storefront_redis" "storefront_mlflow_postgres" "storefront_airflow_postgres")
for service in "${services[@]}"; do
    if docker ps | grep -q "$service"; then
        print_success "$service is running ✓"
    else
        print_warning "$service might not be running properly"
    fi
done

echo
print_success "🎉 Wasaa Storefront ML Stack setup complete!"
echo
print_status "Database Connection Info:"
echo "  - Connects to: storefront_postgres (main Storefront DB)"
echo "  - Database: storefront_db"
echo "  - User: storefront_user"
echo
echo "Available services:"
echo "  - Airflow UI: http://localhost:8081 (admin/admin_secure_2024)"
echo "  - MLflow UI: http://localhost:5000"
echo "  - FastAPI: http://localhost:8000"
echo "  - Redis: localhost:6380"
echo
print_warning "NOTE: This ML system requires the main Storefront microservice to be running"
print_warning "Make sure 'storefront_postgres' container is accessible on the storefront-network"
echo
echo "To view logs: docker-compose logs -f [service-name]"
echo "To stop: docker-compose down"
echo "To restart: docker-compose restart [service-name]"
echo

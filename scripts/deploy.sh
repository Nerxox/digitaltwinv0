#!/bin/bash

# Digital Twin Energy - Production Deployment Script
# This script handles the complete deployment process for production

set -euo pipefail

# Configuration
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
NAMESPACE="digital-twin"
REGISTRY="ghcr.io"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if required environment variables are set
    if [[ -z "${SECRET_KEY:-}" ]]; then
        log_error "SECRET_KEY environment variable is not set"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Backup existing data
backup_data() {
    log_info "Creating backup of existing data..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    if docker-compose -f docker-compose.prod.yml ps db | grep -q "Up"; then
        log_info "Backing up database..."
        docker-compose -f docker-compose.prod.yml exec -T db pg_dump -U energy energydb > "$BACKUP_DIR/database.sql"
    fi
    
    # Backup models and artifacts
    if [[ -d "artifacts" ]]; then
        log_info "Backing up artifacts..."
        cp -r artifacts "$BACKUP_DIR/"
    fi
    
    log_info "Backup completed: $BACKUP_DIR"
}

# Deploy application
deploy_application() {
    log_info "Deploying Digital Twin Energy application..."
    
    # Pull latest images
    log_info "Pulling latest images..."
    docker-compose -f docker-compose.prod.yml pull
    
    # Stop existing services
    log_info "Stopping existing services..."
    docker-compose -f docker-compose.prod.yml down --remove-orphans
    
    # Start services
    log_info "Starting services..."
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    check_service_health
}

# Check service health
check_service_health() {
    log_info "Checking service health..."
    
    # Check database
    if ! docker-compose -f docker-compose.prod.yml exec -T db pg_isready -U energy -d energydb; then
        log_error "Database is not ready"
        exit 1
    fi
    
    # Check Redis
    if ! docker-compose -f docker-compose.prod.yml exec -T redis redis-cli ping; then
        log_error "Redis is not ready"
        exit 1
    fi
    
    # Check backend API
    if ! curl -f http://localhost:8000/healthz > /dev/null 2>&1; then
        log_error "Backend API is not ready"
        exit 1
    fi
    
    log_info "All services are healthy"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Wait for database to be ready
    sleep 10
    
    # Run migrations (if any)
    docker-compose -f docker-compose.prod.yml exec backend python -c "
        from backend_ai.common.db import init_db
        init_db()
    "
    
    log_info "Database migrations completed"
}

# Initialize models
initialize_models() {
    log_info "Initializing ML models..."
    
    # Check if models directory exists
    if [[ ! -d "models" ]]; then
        log_warn "Models directory not found. Please ensure models are trained and available."
        return
    fi
    
    # Copy models to container
    docker-compose -f docker-compose.prod.yml exec backend mkdir -p /app/models
    docker cp models/. digital_twin_backend:/app/models/
    
    log_info "Models initialized"
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Test API endpoints
    if ! curl -f http://localhost:8000/healthz; then
        log_error "Health check failed"
        exit 1
    fi
    
    if ! curl -f http://localhost:8000/readyz; then
        log_error "Readiness check failed"
        exit 1
    fi
    
    # Test API documentation
    if ! curl -f http://localhost:8000/docs; then
        log_error "API documentation not accessible"
        exit 1
    fi
    
    log_info "Smoke tests passed"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Check if Prometheus is running
    if ! curl -f http://localhost:9090 > /dev/null 2>&1; then
        log_warn "Prometheus is not accessible"
    else
        log_info "Prometheus is running"
    fi
    
    # Check if Grafana is running
    if ! curl -f http://localhost:3000 > /dev/null 2>&1; then
        log_warn "Grafana is not accessible"
    else
        log_info "Grafana is running"
    fi
}

# Cleanup old resources
cleanup() {
    log_info "Cleaning up old resources..."
    
    # Remove old images
    docker image prune -f
    
    # Remove old containers
    docker container prune -f
    
    log_info "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting Digital Twin Energy deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    
    check_prerequisites
    backup_data
    deploy_application
    run_migrations
    initialize_models
    run_smoke_tests
    setup_monitoring
    cleanup
    
    log_info "Deployment completed successfully!"
    log_info "Access points:"
    log_info "  - API: http://localhost:8000"
    log_info "  - API Docs: http://localhost:8000/docs"
    log_info "  - Prometheus: http://localhost:9090"
    log_info "  - Grafana: http://localhost:3000"
}

# Handle script arguments
case "${1:-}" in
    "rollback")
        log_info "Rolling back to previous version..."
        # Add rollback logic here
        ;;
    "status")
        log_info "Checking deployment status..."
        docker-compose -f docker-compose.prod.yml ps
        ;;
    "logs")
        log_info "Showing application logs..."
        docker-compose -f docker-compose.prod.yml logs -f
        ;;
    *)
        main
        ;;
esac
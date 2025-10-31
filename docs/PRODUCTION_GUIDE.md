# Digital Twin Energy - Production Guide

## Overview

This guide provides comprehensive instructions for deploying and maintaining the Digital Twin Energy system in a production environment.

## Architecture

The Digital Twin Energy system consists of:

- **Backend API**: FastAPI-based REST API with ML model serving
- **Database**: TimescaleDB for time-series data storage
- **Cache**: Redis for caching and rate limiting
- **Message Broker**: MQTT for real-time data ingestion
- **Monitoring**: Prometheus + Grafana for observability
- **Load Balancer**: Nginx for reverse proxy and SSL termination

## Prerequisites

### System Requirements

- **CPU**: Minimum 4 cores, Recommended 8+ cores
- **Memory**: Minimum 8GB RAM, Recommended 16GB+ RAM
- **Storage**: Minimum 100GB SSD, Recommended 500GB+ SSD
- **Network**: Stable internet connection with low latency

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- Git
- curl (for health checks)

## Security Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Security
SECRET_KEY=your-super-secret-key-change-this-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=postgresql://energy:energy@db:5432/energydb
POSTGRES_PASSWORD=your-secure-database-password

# Redis
REDIS_URL=redis://redis:6379/0
REDIS_PASSWORD=your-secure-redis-password

# MQTT
MQTT_BROKER=mosquitto://mqtt:1883
MQTT_USERNAME=your-mqtt-username
MQTT_PASSWORD=your-mqtt-password

# Monitoring
GRAFANA_PASSWORD=your-grafana-admin-password
PROMETHEUS_RETENTION=200h

# API Configuration
API_RATE_LIMIT=100
API_TIMEOUT=30
```

### SSL/TLS Configuration

1. Obtain SSL certificates from a trusted CA
2. Place certificates in `nginx/ssl/` directory
3. Update `nginx/nginx.conf` with certificate paths

## Deployment

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd digitaltwin

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Deploy with production configuration
./scripts/deploy.sh production
```

### Manual Deployment

1. **Start Infrastructure Services**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d db redis mqtt
   ```

2. **Wait for Services**:
   ```bash
   # Wait for database
   docker-compose -f docker-compose.prod.yml exec db pg_isready -U energy -d energydb
   
   # Wait for Redis
   docker-compose -f docker-compose.prod.yml exec redis redis-cli ping
   ```

3. **Deploy Backend**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d backend
   ```

4. **Deploy Monitoring**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d prometheus grafana
   ```

5. **Deploy Load Balancer**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d nginx
   ```

### Health Checks

Verify deployment:

```bash
# Check all services
docker-compose -f docker-compose.prod.yml ps

# Test API health
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz

# Test API documentation
curl http://localhost:8000/docs
```

## Configuration

### Database Configuration

The system uses TimescaleDB with the following optimizations:

- **Shared Buffers**: 256MB
- **Effective Cache Size**: 1GB
- **Max Connections**: 200
- **Checkpoint Completion**: 90%

### Redis Configuration

Redis is configured with:

- **Max Memory**: 256MB
- **Eviction Policy**: allkeys-lru
- **Persistence**: AOF enabled

### API Configuration

- **Rate Limiting**: 100 requests/minute per IP
- **Request Timeout**: 30 seconds
- **Max Request Size**: 10MB

## Monitoring

### Prometheus Metrics

Available metrics:

- `http_requests_total`: HTTP request count
- `http_request_duration_seconds`: Request duration
- `predictions_total`: Prediction count by algorithm
- `system_cpu_usage_percent`: CPU usage
- `system_memory_usage_bytes`: Memory usage

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/your-password)

Pre-configured dashboards:
- System Overview
- API Performance
- ML Model Performance
- Database Metrics

### Alerting

Configured alerts:

- High error rate (>10% 5xx responses)
- High response time (>2s 95th percentile)
- High CPU usage (>80%)
- High memory usage (>4GB)
- Service downtime

## Maintenance

### Backup Strategy

1. **Database Backup**:
   ```bash
   docker-compose -f docker-compose.prod.yml exec db pg_dump -U energy energydb > backup.sql
   ```

2. **Model Backup**:
   ```bash
   tar -czf models_backup.tar.gz models/
   ```

3. **Configuration Backup**:
   ```bash
   tar -czf config_backup.tar.gz docker-compose.prod.yml .env nginx/
   ```

### Log Management

Logs are stored in `/app/logs/` and rotated daily:

- Application logs: `digital_twin.log`
- Access logs: `access.log`
- Error logs: `error.log`

### Model Updates

To update ML models:

1. Train new models
2. Place models in `models/` directory
3. Restart backend service:
   ```bash
   docker-compose -f docker-compose.prod.yml restart backend
   ```

### Scaling

#### Horizontal Scaling

To scale the backend:

```bash
# Scale backend to 3 replicas
docker-compose -f docker-compose.prod.yml up -d --scale backend=3
```

#### Vertical Scaling

Update resource limits in `docker-compose.prod.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '4.0'
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**:
   - Check database credentials
   - Verify database is running
   - Check network connectivity

2. **Model Loading Failed**:
   - Verify model files exist
   - Check file permissions
   - Review model compatibility

3. **High Memory Usage**:
   - Check for memory leaks
   - Adjust model cache size
   - Scale horizontally

4. **API Timeouts**:
   - Check database performance
   - Review model inference time
   - Adjust timeout settings

### Log Analysis

```bash
# View application logs
docker-compose -f docker-compose.prod.yml logs -f backend

# View database logs
docker-compose -f docker-compose.prod.yml logs -f db

# View all logs
docker-compose -f docker-compose.prod.yml logs -f
```

### Performance Tuning

1. **Database Optimization**:
   - Create appropriate indexes
   - Tune PostgreSQL parameters
   - Use connection pooling

2. **Model Optimization**:
   - Use model quantization
   - Implement model caching
   - Batch predictions

3. **API Optimization**:
   - Enable response compression
   - Use HTTP/2
   - Implement caching

## Security Best Practices

1. **Authentication**:
   - Use strong passwords
   - Implement JWT token rotation
   - Enable multi-factor authentication

2. **Network Security**:
   - Use HTTPS only
   - Implement firewall rules
   - Use VPN for admin access

3. **Data Protection**:
   - Encrypt sensitive data
   - Regular security updates
   - Monitor access logs

## Disaster Recovery

### Recovery Procedures

1. **Database Recovery**:
   ```bash
   # Restore from backup
   docker-compose -f docker-compose.prod.yml exec -T db psql -U energy -d energydb < backup.sql
   ```

2. **Full System Recovery**:
   ```bash
   # Stop all services
   docker-compose -f docker-compose.prod.yml down
   
   # Restore from backup
   tar -xzf full_backup.tar.gz
   
   # Restart services
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Backup Schedule

- **Database**: Daily at 2 AM
- **Models**: Weekly on Sunday
- **Configuration**: Monthly
- **Full System**: Quarterly

## Support

For production support:

1. Check logs first
2. Review monitoring dashboards
3. Consult this documentation
4. Contact support team

## API Documentation

The API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI Spec: `http://localhost:8000/openapi.json`

## Performance Benchmarks

Expected performance metrics:

- **API Response Time**: <200ms (95th percentile)
- **Prediction Latency**: <500ms
- **Throughput**: 1000+ requests/minute
- **Uptime**: 99.9%

## Compliance

The system is designed to meet:

- **GDPR**: Data protection and privacy
- **ISO 27001**: Information security
- **SOC 2**: Security and availability
- **HIPAA**: Healthcare data protection (if applicable)
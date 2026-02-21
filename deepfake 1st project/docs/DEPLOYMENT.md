# Deployment Guide

## Overview

This guide covers deploying the Deepfake Detector to production environments.

## Pre-Deployment Checklist

- [ ] Model weights obtained (train or download)
- [ ] Environment variables configured
- [ ] HTTPS/SSL certificate (if using domain)
- [ ] Database setup (PostgreSQL for production)
- [ ] Monitoring/logging configured
- [ ] Rate limiting enabled
- [ ] Security headers added
- [ ] CORS properly configured

## Local Deployment

See SETUP.md

## Docker Deployment

### Build Images

```bash
# Backend
docker build -f docker/Dockerfile.backend \
  -t deepfake-detector-api:latest .

# Frontend
docker build -f docker/Dockerfile.frontend \
  -t deepfake-detector-web:latest .

# Or use Docker Compose
docker-compose -f docker/docker-compose.yml build
```

### Run Containers

```bash
# Using Docker Compose (Recommended)
docker-compose -f docker/docker-compose.yml up -d

# Manual Docker commands
docker run -d \
  -p 8000:8000 \
  -e DEBUG=False \
  -e DATABASE_URL=postgresql://... \
  -v weights:/app/weights \
  deepfake-detector-api:latest

docker run -d \
  -p 3000:3000 \
  -e REACT_APP_API_URL=https://api.yourdomain.com \
  deepfake-detector-web:latest
```

### Environment Variables (Production)

```
DEBUG=False
HOST=0.0.0.0
PORT=8000

MODEL_NAME=efficientnet-b0
MODEL_PATH=weights/deepfake_model.pth
DEVICE=cuda

MAX_FILE_SIZE_MB=50
UPLOAD_DIR=/data/uploads

DATABASE_URL=postgresql://user:pass@db.example.com:5432/deepfake

CORS_ORIGINS=["https://yourdomain.com", "https://www.yourdomain.com"]

LOG_LEVEL=WARNING
```

## AWS Deployment

### EC2 Setup

```bash
# 1. Launch instance
# - AMI: Ubuntu 22.04 LTS
# - Type: t3.large (for CPU) or g4dn.xlarge (for GPU)
# - Storage: 100GB gp3
# - Security Group: Allow 80, 443, 22

# 2. SSH in
ssh -i "key.pem" ubuntu@<INSTANCE_IP>

# 3. Update system
sudo apt update && sudo apt upgrade -y

# 4. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 5. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 6. Clone project
git clone https://github.com/yourrepo/deepfake-detector.git
cd deepfake-detector

# 7. Create environment
cp backend/.env.example backend/.env
nano backend/.env  # Edit settings

# 8. Start services
docker-compose -f docker/docker-compose.yml up -d

# 9. Verify
curl http://localhost:8000/health
```

### RDS Database Setup

```bash
# 1. Create RDS instance
# - Engine: PostgreSQL 14+
# - Instance: db.t3.micro (dev) or db.t3.small (prod)
# - Storage: 100GB gp3
# - Multi-AZ: Yes

# 2. Get endpoint
# RDS console → Databases → your-db → Endpoint

# 3. Update .env
DATABASE_URL=postgresql://admin:password@deepfake-db.c7.us-east-1.rds.amazonaws.com:5432/deepfake_db

# 4. Initialize database
docker exec backend python -c "from app.utils.database import init_db; init_db()"
```

### Elastic Load Balancer (ALB)

```bash
# 1. Create Application Load Balancer
# - Name: deepfake-alb
# - Scheme: Internet-facing
# - Subnets: Select both AZs

# 2. Create Target Groups
# Backend: Port 8000, Health: /health
# Frontend: Port 3000

# 3. Add Listeners
# HTTP → redirect to HTTPS
# HTTPS → forward to targets

# 4. Map domains
# yourdomain.com → Frontend
# api.yourdomain.com → Backend
```

### Auto Scaling Group

```bash
# 1. Create Launch Template
# - Image: Ubuntu 22.04
# - Instance type: t3.large
# - User Data: (see above)

# 2. Create ASG
# - Min: 2
# - Desired: 3
# - Max: 10
# - Health Check: ELB

# 3. Scaling Policies
# CPU > 70% → scale up
# CPU < 30% → scale down
```

### SSL Certificate (ACM)

```bash
# 1. Request certificate in AWS Certificate Manager
# - Domain: yourdomain.com, www.yourdomain.com
# - Validation: DNS

# 2. Add CNAME records to DNS

# 3. Attach to ALB listener
```

## Railway Deployment

### Backend

```bash
# 1. Create Railway project
railway init

# 2. Add environment variable
railway variables set DATABASE_URL=postgresql://...
railway variables set DEBUG=False

# 3. Create railway.json
cat > railway.json << EOF
{
  "build": {
    "builder": "dockerfile",
    "dockerfilePath": "docker/Dockerfile.backend"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0"
  }
}
EOF

# 4. Deploy
railway up
```

### Frontend

```bash
# 1. Set environment
railway variables set REACT_APP_API_URL=https://backend-url.railway.app

# 2. Create railway.json
cat > railway.json << EOF
{
  "build": {
    "builder": "dockerfile",
    "dockerfilePath": "docker/Dockerfile.frontend"
  }
}
EOF

# 3. Deploy
railway up
```

## Render Deployment

### Backend

1. Connect GitHub repo to Render
2. Create **Web Service**
3. Set:
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Add DATABASE_URL, DEBUG, etc.
4. Deploy

### Frontend

1. Create **Static Site**
2. Set:
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Publish Directory**: `frontend/build`
   - **Environment**: REACT_APP_API_URL
3. Deploy

## Nginx Configuration

```nginx
# /etc/nginx/sites-available/deepfake
upstream backend {
    server 127.0.0.1:8000;
}

upstream frontend {
    server 127.0.0.1:3000;
}

# HTTP → HTTPS redirect
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_types text/plain text/css text/xml text/javascript 
               application/x-javascript application/xml+rss 
               application/json image/svg+xml;

    # Frontend
    location / {
        proxy_pass http://frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # API
    location /api/ {
        proxy_pass http://backend/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # API docs
    location /docs {
        proxy_pass http://backend/docs;
    }

    location /openapi.json {
        proxy_pass http://backend/openapi.json;
    }
}
```

## SSL/TLS with Let's Encrypt

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get certificate
sudo certbot certonly --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer

# Test renewal
sudo certbot renew --dry-run
```

## Monitoring & Logging

### CloudWatch (AWS)

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configure
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard

# Start
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config -m ec2 \
  -s -c file:/opt/aws/amazon-cloudwatch-agent/etc/cloudwatch-config.json
```

### Application Performance Monitoring

```python
# Sentry
import sentry_sdk

sentry_sdk.init(
    dsn="https://...@sentry.io/...",
    traces_sample_rate=0.1,
    environment="production"
)

# New Relic
pip install newrelic
NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program uvicorn main:app
```

### Log Aggregation (ELK Stack)

```bash
# Docker Compose with ELK
docker-compose -f docker/docker-compose.elk.yml up -d

# Kibana: http://localhost:5601
```

## Performance Tuning

### Backend

```python
# Requirements for 1000+ concurrent users:

# 1. Use Gunicorn with multiple workers
# workers = 4 * CPU_cores

# 2. Connection pooling
from sqlalchemy.pool import QueuePool
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40
)

# 3. Redis caching
from redis import Redis
cache = Redis(host='localhost', port=6379, decode_responses=True)

# 4. Model quantization
model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### Frontend

```bash
# Build optimization
npm run build

# Code splitting
import React, { Suspense, lazy } from 'react';
const Results = lazy(() => import('./components/Results'));

# Caching
# Cache-Control: max-age=31536000 for assets
# Service Workers for offline support
```

### Database

```sql
-- Indexes
CREATE INDEX idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX idx_predictions_prediction ON predictions(prediction);

-- Connection pooling
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB

-- Replication (for HA)
-- Primary-replica setup for read scaling
```

## Backup & Recovery

```bash
# Database backups
pg_dump -U postgres -h db.example.com deepfake_db > backup.sql

# Automated daily backups
0 2 * * * pg_dump -U postgres -h db.example.com deepfake_db | gzip > /backups/db-$(date +%Y%m%d).sql.gz

# S3 backup
aws s3 cp backup.sql s3://deepfake-backups/$(date +%Y%m%d).sql

# Restore
psql -U postgres -h db.example.com deepfake_db < backup.sql
```

## Disaster Recovery Plan

| Component | RPO | RTO | Strategy |
|-----------|-----|-----|----------|
| API Server | 15min | 2min | ASG failover |
| Database | 5min | 10min |Automated backups, replicas |
| Uploads | 1h | 30min | S3 replication |
| Code | - | 5min | Blue-green deployment |

## Cost Optimization

### Estimated Monthly Costs (AWS)

| Component | Usage | Cost |
|-----------|-------|------|
| EC2 (t3.large) | 2 instances | $60 |
| RDS (db.t3.small) | Single AZ | $40 |
| ALB | 1 LB | $16 |
| Data transfer | 1TB | $100 |
| **Total** | | **~$216** |

**Cost reduction strategies**:
1. Spot instances (70% cheaper)
2. Reserved instances (40% discount)
3. Smaller instance types during off-hours
4. Compress images aggressively
5. CDN for static assets

---

**Deployment Complete!** 🚀

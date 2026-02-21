# Setup & Installation Guide

## Prerequisites

Before starting, ensure you have:

- **Python 3.9+** (check with `python --version`)
- **Node.js 16+** (check with `node --version`)
- **Git** (optional, for cloning)
- **Docker & Docker Compose** (for containerized setup)
- **CUDA 11.8+** (optional, for GPU acceleration)

## Option 1: Quick Start with Docker (Recommended)

### Setup

```bash
# 1. Navigate to project directory
cd deepfake-detector

# 2. Start services with Docker Compose
docker-compose -f docker/docker-compose.yml up --build

# First time may take 5-10 minutes to build images
```

### Access

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Stop Services

```bash
docker-compose -f docker/docker-compose.yml down
```

---

## Option 2: Local Development Setup

### Backend Setup

#### 1. Create Virtual Environment

```bash
cd backend

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installation
pip freeze | grep torch
```

#### 3. Create Environment File

```bash
# Copy template
cp .env.example .env

# Edit .env with your settings
# Change DEBUG, DATABASE_URL, etc.
```

#### 4. Create Required Directories

```bash
mkdir -p uploads
mkdir -p weights
```

#### 5. Run Backend Server

```bash
# Option 1: Using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Option 2: Using Python main
python main.py

# You should see:
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test Backend**:
```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy", ...}
```

---

### Frontend Setup

#### 1. Install Node Dependencies

```bash
cd frontend
npm install

# This creates node_modules/ and package-lock.json
# Takes 2-5 minutes
```

#### 2. Configure Environment

```bash
# Create .env file
cat > .env << EOF
REACT_APP_API_URL=http://localhost:8000
EOF
```

#### 3. Start Development Server

```bash
npm start

# Should automatically open http://localhost:3000 in browser
# Hot reload enabled - changes reflect immediately
```

**Terminal Output**:
```
Compiled successfully!

You can now view deepfake-detector in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

---

## Option 3: Production Deployment

### Prerequisites
- Linux server (Ubuntu 22.04+)
- Docker & Docker Compose installed
- Domain name (optional)
- SSL certificate (optional)

### Deployment Steps

```bash
# 1. SSH into server
ssh -i "key.pem" ubuntu@your-server-ip

# 2. Update system
sudo apt update && sudo apt upgrade -y

# 3. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 4. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 5. Clone project
git clone https://github.com/yourrepo/deepfake-detector.git
cd deepfake-detector

# 6. Create .env file
cp backend/.env.example backend/.env

# Edit production settings
nano backend/.env
# Set: DEBUG=False, DATABASE_URL=postgresql://..., etc.

# 7. Build and start
docker-compose -f docker/docker-compose.yml up -d --build

# 8. Verify
curl http://localhost:8000/health
curl http://localhost:3000
```

### With Nginx Reverse Proxy

```bash
# Install Nginx
sudo apt install nginx -y

# Create config
sudo cat > /etc/nginx/sites-available/deepfake << EOF
upstream backend {
    server localhost:8000;
}

upstream frontend {
    server localhost:3000;
}

server {
    listen 80;
    server_name yourdomain.com;

    # Frontend
    location / {
        proxy_pass http://frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
    }

    # API
    location /api {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/deepfake /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### With SSL (Let's Encrypt)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Generate certificate
sudo certbot certonly --nginx -d yourdomain.com

# Auto-renew
sudo systemctl enable certbot.timer
```

---

## Configuration

### Backend Configuration (.env)

```
# App settings
DEBUG=False
HOST=0.0.0.0
PORT=8000

# Model
MODEL_NAME=efficientnet-b0
MODEL_PATH=weights/deepfake_model.pth
DEVICE=cuda

# File upload
MAX_FILE_SIZE_MB=10
UPLOAD_DIR=uploads

# Database
DATABASE_URL=sqlite:///./predictions.db
# Production: postgresql://user:password@host/db

# CORS
CORS_ORIGINS=["http://localhost:3000", "https://yourdomain.com"]

# Logging
LOG_LEVEL=INFO
```

### Frontend Configuration (.env)

```
# API endpoint
REACT_APP_API_URL=http://localhost:8000
# Production: https://api.yourdomain.com
```

---

## Verification

### Backend Verification

```bash
# 1. Check server is running
curl http://localhost:8000/

# 2. Check health
curl http://localhost:8000/health

# 3. Check API docs
# Open: http://localhost:8000/docs

# 4. Check model info
curl http://localhost:8000/model-info

# 5. Test prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.jpg"
```

### Frontend Verification

```bash
# 1. Check it's running
curl http://localhost:3000

# 2. Open in browser
# http://localhost:3000

# 3. Try uploading an image
# (you'll need a test image file)
```

---

## Troubleshooting

### Issue: Port Already in Use

```bash
# Find process on port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Or use different port
uvicorn main:app --port 8001
```

### Issue: Module 'torch' Not Found

```bash
# Verify virtual environment is active
which python  # Should show venv path

# Reinstall torch
pip uninstall torch -y
pip install torch torchvision

# Or with GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: CORS Errors in Frontend

```bash
# Backend: Edit app/config.py
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    "*"  # Allow all (dev only!)
]

# Restart backend
```

### Issue: npm install taking too long

```bash
# Clear npm cache
npm cache clean --force

# Use different registry
npm install --registry https://registry.npmjs.org/

# Or use yarn
npm install -g yarn
yarn install
```

### Issue: Docker builds timeout

```bash
# Increase timeout
docker-compose -f docker/docker-compose.yml up --build --no-cache

# Or build separately
docker build -f docker/Dockerfile.backend -t deepfake-backend:latest .
docker build -f docker/Dockerfile.frontend -t deepfake-frontend:latest .
```

### Issue: GPU not being used

```bash
# Check CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False:
# 1. Install NVIDIA drivers
# 2. Install CUDA toolkit
# 3. Install cuDNN

# Set device to CPU in .env
DEVICE=cpu
```

---

## Performance Optimization

### Backend Optimization

```python
# Use Gunicorn for production
pip install gunicorn

# Run with multiple workers
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker

# Or Docker
ENTRYPOINT ["gunicorn", "main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]
```

### Frontend Optimization

```bash
# Create production build
npm run build

# This creates optimized dist/ folder
# ~500KB gzipped

# Serve with compression
npm install -g serve
serve -s build -l 3000 -c ../ecosystem.config.js
```

### Database Optimization (PostgreSQL)

```bash
# Create index on predictions
CREATE INDEX idx_predictions_timestamp ON predictions(created_at DESC);
CREATE INDEX idx_predictions_prediction ON predictions(prediction);

# Connection pooling
# Use pgBouncer or connection pool
```

---

## Monitoring

### Logs

```bash
# Backend logs
tail -f backend/app.log

# Frontend console
# Open browser DevTools: F12 → Console

# Docker logs
docker logs container_name
docker-compose -f docker/docker-compose.yml logs -f
```

### Metrics

```bash
# Monitor resource usage
docker stats

# System monitoring
top  # macOS/Linux
tasklist  # Windows
```

---

## Next Steps

1. **Download Pretrained Model**: 
   - Place `deepfake_model.pth` in `weights/` directory
   - Or train your own (see TRAINING.md)

2. **Test the Application**:
   - Upload test image with UI
   - Check API responses
   - Verify database logging

3. **Customize**:
   - Change model architecture
   - Adjust preprocessing
   - Modify UI styling

4. **Deploy**:
   - Follow production deployment steps
   - Configure domain & SSL
   - Monitor performance

---

## Getting Help

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@deepfakedetector.com
- **Documentation**: Full docs in `/docs` folder

---

**Setup Completed!** 🎉

You should now have:
- ✅ Backend API running on http://localhost:8000
- ✅ Frontend running on http://localhost:3000
- ✅ Database initialized
- ✅ Ready for image uploads

Next: **Upload a test image and verify predictions!**

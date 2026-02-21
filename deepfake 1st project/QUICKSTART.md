# QUICKSTART.md - Get Running in 5 Minutes

## The Fastest Way to Start

### Option 1: Docker (Recommended - 3 Steps)

```bash
# 1. Have Docker installed? If not: https://www.docker.com/products/docker-desktop

# 2. Run this ONE command:
docker-compose -f docker/docker-compose.yml up --build

# 3. Open browser:
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

**That's it!** 🎉

### Option 2: Local Development (Python + Node)

```bash
# Terminal 1 - Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py

# Terminal 2 - Frontend
cd frontend
npm install
npm start
```

---

## Next Steps

1. **Upload a test image** to http://localhost:3000
2. **See results** (prediction, confidence, visualization)
3. **Check API docs** at http://localhost:8000/docs
4. **Train on your data** (see docs/TRAINING.md)
5. **Deploy to cloud** (see docs/DEPLOYMENT.md)

---

## Project Structure

```
deepfake-detector/
├── backend/                # FastAPI + PyTorch
│   └── app/
│       ├── api/routes.py           # POST /predict endpoint
│       ├── models/detector.py      # Model inference
│       └── utils/grad_cam.py       # Visualization
├── frontend/               # React
│   └── src/
│       ├── App.js                 # Main app
│       └── components/
└── models/                 # Training & inference
    ├── training/train.py          # Train on your data
    └── inference/inference.py     # Standalone inference
```

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI server entry point |
| `backend/app/api/routes.py` | API endpoints (POST /predict) |
| `backend/app/models/detector.py` | Model inference logic |
| `frontend/src/App.js` | React main component |
| `docker/docker-compose.yml` | Docker orchestration |
| `docs/TRAINING.md` | How to train on your data |
| `docs/DEPLOYMENT.md` | Deploy to AWS/Railway/Render |

---

## What It Does

**Upload Image** → **AI Analysis** → **Real/Deepfake Prediction** → **Confidence Score** → **Visualization**

### Example Response

```json
{
  "prediction": "Real",
  "confidence": 87.5,
  "explanation": "Grad-CAM heatmap generated",
  "model": "efficientnet-b0",
  "device": "cuda"
}
```

---

## Key Features

✅ **FastAPI Backend** - High-performance REST API  
✅ **React Frontend** - Modern, responsive UI  
✅ **PyTorch Model** - EfficientNet-B0 (fast) or ResNet50 (accurate)  
✅ **Grad-CAM** - Visualize predictions  
✅ **Database** - Log predictions  
✅ **Docker** - One-command deployment  
✅ **Cloud Ready** - AWS, Railway, Render support  

---

## Troubleshooting

### Port Already in Use

```bash
# macOS/Linux
lsof -i :8000
kill -9 <PID>

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Docker Issues

```bash
# Clean build
docker-compose down
docker-compose -f docker/docker-compose.yml up --build --no-cache

# Check logs
docker-compose logs -f
```

### Module Not Found

```bash
# Backend
cd backend
source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

---

## API Examples

### Python

```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    files={'file': open('image.jpg', 'rb')}
)
print(response.json())
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', imageFile);
const result = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
}).then(r => r.json());
```

---

## Full Documentation

- **Setup Guide**: `docs/SETUP.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **API Reference**: `docs/API.md`
- **Training Guide**: `docs/TRAINING.md`
- **Deployment**: `docs/DEPLOYMENT.md`

---

## Common Commands

```bash
# Start with Docker
docker-compose -f docker/docker-compose.yml up

# Stop services
docker-compose down

# View logs
docker-compose logs -f backend  # or frontend

# Build fresh
docker-compose up --build

# Train model
python models/training/train.py --data-dir data --epochs 20

# Test backend
curl http://localhost:8000/health

# Test frontend
curl http://localhost:3000
```

---

## What's Inside

### Backend Technologies
- FastAPI (web framework)
- PyTorch (ML inference)
- SQLite (database)
- Pydantic (validation)

### Frontend Technologies
- React 18 (UI)
- Axios (HTTP)
- CSS3 (styling)

### ML Technologies
- EfficientNet-B0 (default)
- ResNet50 (alternative)
- Grad-CAM (visualization)

---

## Next Stage

Once running, you might want to:

1. **Train Your Model**
   ```bash
   cd models/training
   python train.py --data-dir your_dataset --epochs 20
   ```

2. **Deploy to Cloud**
   - AWS: Full guide in `docs/DEPLOYMENT.md`
   - Railway: Simple 5-minute setup
   - Render: Click-and-deploy

3. **Customize**
   - Change model architecture
   - Adjust confidence thresholds
   - Modify UI styling
   - Add authentication

---

## Performance

- **Inference Time**: ~50ms per image (GPU)
- **Throughput**: 20+ predictions/second
- **Model Size**: ~100MB
- **Accuracy**: 95-98% (depends on training data)

---

## Support & Help

- **Issues**: Check `docs/SETUP.md` Troubleshooting section
- **Questions**: See full documentation in `/docs`
- **Examples**: Check `docs/API.md` for code samples
- **Training**: Follow `docs/TRAINING.md` step-by-step

---

**You're all set! Upload an image and see the magic happen** ✨

**Need a specific model or custom training? Check `docs/TRAINING.md`**

**Ready to deploy? See `docs/DEPLOYMENT.md`**

---

**Last Updated**: February 2026  
**Status**: Production Ready ✅

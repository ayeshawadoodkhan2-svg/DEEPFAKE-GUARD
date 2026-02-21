# API Documentation

## Overview

The Deepfake Detector API is a RESTful service built with FastAPI. All endpoints return JSON responses.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://api.deepfakedetector.com` (example)

## Authentication

Currently, the API does not require authentication. For production, implement JWT tokens:

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@router.post("/predict")
async def predict(file: UploadFile, credentials = Security(security)):
    # Validate JWT token
    pass
```

## Endpoints

### 1. POST /predict

Analyze an image to determine if it's a deepfake or authentic.

**Request**:

```http
POST /predict HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data

Content: [binary image data]
```

**Parameters**:
- `file` (required, multipart/form-data): Image file (JPG or PNG)

**Constraints**:
- Maximum file size: 10 MB
- Supported formats: JPEG, PNG
- Minimum resolution: 64×64 pixels
- Maximum resolution: 8192×8192 pixels

**Response** (200 OK):

```json
{
    "prediction": "Real",
    "confidence": 87.5,
    "explanation": "Grad-CAM heatmap generated",
    "filename": "photo.jpg",
    "model": "efficientnet-b0",
    "device": "cuda",
    "grad_cam_available": true
}
```

**Response Fields**:
- `prediction` (string): "Real" or "Deepfake"
- `confidence` (float): Confidence score 0-100%
- `explanation` (string): Human-readable explanation
- `filename` (string): Original filename
- `model` (string): Model used for prediction
- `device` (string): "cuda" or "cpu"
- `grad_cam_available` (boolean): Whether visualization is available

**Error Responses**:

```json
{
    "detail": "Invalid file type. Allowed: ['jpg', 'jpeg', 'png']"
}
```

| Status Code | Reason |
|------------|--------|
| 400 | Invalid file type or missing file |
| 413 | File size exceeds limit |
| 422 | Unprocessable entity (validation error) |
| 500 | Server error during processing |

**cURL Example**:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@photo.jpg"
```

**Python Example**:

```python
import requests

url = "http://localhost:8000/predict"
files = {'file': open('photo.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

**JavaScript Example**:

```javascript
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(result);
```

---

### 2. GET /health

Health check endpoint to verify API is running.

**Request**:

```http
GET /health HTTP/1.1
Host: localhost:8000
```

**Response** (200 OK):

```json
{
    "status": "healthy",
    "app": "Deepfake Detector API",
    "version": "1.0.0"
}
```

**cURL Example**:

```bash
curl http://localhost:8000/health
```

**Use Case**: Monitoring, load balancers, health probes

---

### 3. GET /model-info

Get information about the loaded model.

**Request**:

```http
GET /model-info HTTP/1.1
Host: localhost:8000
```

**Response** (200 OK):

```json
{
    "model_name": "efficientnet-b0",
    "device": "cuda",
    "model_path": "weights/deepfake_model.pth",
    "parameters": {
        "total_parameters": 4010000,
        "trainable_parameters": 4010000
    }
}
```

**cURL Example**:

```bash
curl http://localhost:8000/model-info
```

---

### 4. GET /

Root endpoint with documentation links.

**Response** (200 OK):

```json
{
    "message": "Deepfake Detector API",
    "version": "1.0.0",
    "docs": "/docs",
    "redoc": "/redoc"
}
```

---

## Interactive API Documentation

### Swagger UI

Access at: `http://localhost:8000/docs`

Features:
- Interactive endpoint testing
- Request/response examples
- Schema definitions
- Model documentation

### ReDoc

Access at: `http://localhost:8000/redoc`

Features:
- Alternative documentation format
- Better mobile experience
- Detailed endpoint descriptions

---

## Error Handling

### Standard Error Response Format

```json
{
    "detail": "Error message describing what went wrong"
}
```

### Common Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Prediction completed |
| 400 | Bad Request | Invalid file type |
| 413 | Payload Too Large | File exceeds 10MB |
| 422 | Unprocessable Entity | Validation error |
| 500 | Server Error | Model inference failed |
| 503 | Service Unavailable | Server overloaded |

### Example Error Handling

```python
try:
    response = requests.post(url, files=files)
    response.raise_for_status()  # Raise for 4xx/5xx
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error {e.response.status_code}: {e.response.json()}")
except requests.exceptions.ConnectionError:
    print("Failed to connect to API")
```

---

## Rate Limiting

Currently not implemented. For production, add:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@router.post("/predict")
@limiter.limit("30/minute")
async def predict(request: Request, file: UploadFile):
    pass
```

---

## CORS Headers

The API includes CORS headers to allow cross-origin requests:

```http
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

Modify in `app/config.py`:

```python
CORS_ORIGINS = ["http://localhost:3000", "https://yourdomain.com"]
```

---

## Batch Processing (Future)

Proposed endpoint for multiple images:

```http
POST /predict-batch HTTP/1.1
Content-Type: multipart/form-data

files=@image1.jpg&files=@image2.jpg&files=@image3.jpg
```

**Response**:

```json
{
    "results": [
        {"filename": "image1.jpg", "prediction": "Real", "confidence": 92.3},
        {"filename": "image2.jpg", "prediction": "Deepfake", "confidence": 88.1},
        {"filename": "image3.jpg", "prediction": "Real", "confidence": 95.6}
    ]
}
```

---

## Webhook Notifications (Future)

Proposed for async processing:

```http
POST /predict-async HTTP/1.1
Content-Type: application/json

{
    "image_url": "https://example.com/image.jpg",
    "webhook_url": "https://yourdomain.com/webhook"
}
```

Webhook callback:

```json
POST https://yourdomain.com/webhook
{
    "task_id": "uuid",
    "status": "completed",
    "result": {
        "prediction": "Real",
        "confidence": 87.5
    }
}
```

---

## Performance Metrics

### Typical Response Times

| Model | Device | Time | Throughput |
|-------|--------|------|-----------|
| EfficientNet-B0 | GPU | 50ms | 20 req/s |
| EfficientNet-B0 | CPU | 150ms | 7 req/s |
| ResNet50 | GPU | 80ms | 12 req/s |

### Concurrent Requests

- Single server: 100-500 concurrent connections
- With Gunicorn workers: 1000+ concurrent connections
- With load balancer: 10,000+ concurrent connections

---

## Best Practices

### 1. Error Handling

```python
try:
    response = requests.post(api_url, files=files, timeout=30)
    if response.status_code == 200:
        result = response.json()
    else:
        print(f"API error: {response.status_code}")
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.ConnectionError:
    print("Connection failed")
```

### 2. Streaming Large Uploads

```python
# For files > 5MB
with open('large_image.jpg', 'rb') as f:
    response = requests.post(
        api_url,
        files={'file': f},
        timeout=60
    )
```

### 3. Caching Results

```python
import hashlib
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_prediction(file_hash):
    # Cache based on file content hash
    return api_call(file_hash)
```

### 4. Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def predict_with_retry(file):
    return requests.post(api_url, files={'file': file})
```

---

## Versioning

Current API version: `1.0.0`

Future versions will be available at:
- `/api/v2/predict` (hypothetical)
- Backwards compatibility maintained for 2 years

---

## Support & Feedback

- **Issues**: https://github.com/yourrepo/issues
- **Email**: support@deepfakedetector.com
- **Docs**: https://docs.deepfakedetector.com

---

Last Updated: February 2026

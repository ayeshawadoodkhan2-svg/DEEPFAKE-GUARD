# PROJECT STRUCTURE & FILE GUIDE

This document provides a complete overview of the project structure and what each file does.

## 📁 Root Directory

```
deepfake-detector/
├── README.md                 # Main project documentation
├── QUICKSTART.md             # Get started in 5 minutes
├── .gitignore                # Git ignore file
│
├── backend/                  # Python/FastAPI backend
├── frontend/                 # React frontend
├── models/                   # ML models and training
├── docker/                   # Docker configuration
├── docs/                     # Detailed documentation
└── weights/                  # Model weights (download or train)
```

---

## 📦 Backend Directory (`backend/`)

```
backend/
├── main.py                   # FastAPI application entry point
│                             # - Initializes app
│                             # - Sets up middleware (CORS, logging)
│                             # - Registers routes
│                             # - Handles startup/shutdown events
│
├── requirements.txt          # Python dependencies
│                             # - FastAPI, PyTorch, Pillow, etc.
│
├── .env.example              # Environment variables template
│                             # Copy to .env and customize
│
└── app/
    ├── __init__.py           # Package initialization
    │
    ├── config.py             # Configuration settings
    │                          # - App name/version
    │                          # - Server settings
    │                          # - Model paths
    │                          # - Database URL
    │                          # - CORS origins
    │
    ├── api/
    │   ├── __init__.py
    │   └── routes.py         # API endpoints
    │       ├── POST /predict      # Main prediction endpoint
    │       ├── GET /health        # Health check
    │       ├── GET /model-info    # Model information
    │       └── GET /               # Root endpoint
    │
    ├── models/
    │   ├── __init__.py
    │   └── detector.py       # Core deepfake detector
    │       ├── DeepfakeDetector class
    │       ├── _build_model()      # Build EfficientNet/ResNet
    │       ├── _load_weights()     # Load pretrained weights
    │       └── predict()           # Make predictions
    │
    └── utils/
        ├── __init__.py
        │
        ├── image_processing.py    # Image preprocessing
        │   ├── preprocess_image()      # Resize, normalize
        │   ├── denormalize_image()     # Reverse preprocessing
        │   └── get_preprocessing_transform()
        │
        ├── database.py           # Database models and ORM
        │   ├── Prediction model   # Database schema
        │   ├── SessionLocal       # DB session
        │   ├── init_db()          # Initialize database
        │   └── Base               # SQLAlchemy base
        │
        └── grad_cam.py           # Grad-CAM visualization
            ├── GradCAM class     # Grad-CAM implementation
            ├── generate_grad_cam()    # Generate heatmap
            └── apply_heatmap()    # Apply to image
```

### Key Backend Files Explained

**`main.py`** - The entry point
- Creates FastAPI app with title/description
- Adds CORS middleware (allows cross-origin requests)
- Includes routes from `api/routes.py`
- Initializes database on startup

**`config.py`** - Configuration management
- Reads from `.env` file using Pydantic Settings
- Defines all configuration values with defaults
- Centralized settings for entire application

**`api/routes.py`** - REST API endpoints
- `POST /predict`: Accepts image, returns prediction
  - Validates file type/size
  - Preprocesses image
  - Runs inference
  - Generates Grad-CAM
  - Saves to database
  - Returns JSON response

**`models/detector.py`** - Model inference
- Loads EfficientNet-B0 or ResNet50
- Fine-tuned for binary classification (Real/Deepfake)
- Handles tensor conversion and inference
- Returns prediction and confidence score

**`utils/image_processing.py`** - Image preprocessing
- Resizes to 224×224 (ImageNet standard)
- Converts to tensor
- Normalizes using ImageNet statistics
- Critical for consistent model input

**`utils/database.py`** - Database layer
- SQLAlchemy ORM models
- Prediction schema with fields:
  - filename, prediction, confidence
  - explanation, image dimensions
  - created_at timestamp
- Automatic table creation

**`utils/grad_cam.py`** - Visualization
- Grad-CAM implementation for interpretability
- Shows which image regions influenced prediction
- Helps debug model decisions
- Builds user trust

---

## 🎨 Frontend Directory (`frontend/`)

```
frontend/
├── package.json              # Node.js dependencies
│                             # - React, Axios, react-icons
│
├── .env                      # Environment variables
│                             # - REACT_APP_API_URL
│
├── public/
│   └── index.html            # HTML entry point
│       └── <div id="root">   # React mounts here
│
├── src/
│   ├── index.js              # React entry point
│   │                          # - Imports App component
│   │                          # - Creates root and renders
│   │
│   ├── index.css             # Global styles
│   │                          # - CSS variables (colors, etc.)
│   │                          # - Layout styles
│   │                          # - Responsive design
│   │
│   ├── App.js                # Main component
│   │                          # - Manages image state
│   │                          # - Handles predictions
│   │                          # - Renders child components
│   │
│   ├── App.css               # App-specific styles
│   │
│   └── components/
│       ├── Header.js         # App header
│       │   └── Title, subtitle, branding
│       │
│       ├── ImageUploader.js  # Image upload
│       │   ├── Drag-and-drop zone
│       │   ├── File selection
│       │   └── Preview display
│       │
│       ├── Results.js        # Results display
│       │   ├── Prediction badge
│       │   ├── Confidence display
│       │   ├── Model info
│       │   └── Explanation text
│       │
│       └── ConfidenceBar.js  # Confidence visualization
│           └── Animated progress bar
│
└── tailwind.config.js        # Tailwind CSS configuration
```

### Key Frontend Files Explained

**`index.js`** - React entry point
- Imports React and ReactDOM
- Creates root element
- Renders App component

**`App.js`** - Main application component
- State management for:
  - Current image
  - Preview URL
  - Predictions
  - Loading/error states
- Handles API communication with `axios`
- Renders different components based on state

**`components/ImageUploader.js`**
- Drag-and-drop functionality
- File input handling
- Displays preview of selected image
- Responsive design

**`components/Results.js`**
- Shows prediction (Real/Deepfake)
- Displays confidence score
- Shows model information
- Explains what result means
- Displays original image

**`components/ConfidenceBar.js`**
- Visual progress bar
- Color-coded by confidence level
  - Red (high confidence)
  - Yellow (medium)
  - Blue/Green (low)
- Smooth animation

**`index.css`** - Global styling
- CSS variables for theming
- Responsive breakpoints
- Component styles
- Color scheme

---

## 🤖 Models Directory (`models/`)

```
models/
├── inference/
│   ├── __init__.py
│   └── inference.py          # Standalone inference script
│       └── predict_image()   # CLI image prediction
│
└── training/
    └── train.py              # Training script
        ├── DeepfakeDataset class   # Dataset loader
        ├── DeepfakeDetectorTrainer # Training loop
        ├── train_epoch()           # One training epoch
        ├── validate()              # Validation loop
        └── save_model()            # Save weights
```

### Training Guide

**Dataset Structure**:
```
data/
├── real/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── deepfake/
    ├── fake1.jpg
    ├── fake2.jpg
    └── ...
```

**Training Command**:
```bash
python models/training/train.py \
  --data-dir data \
  --epochs 20 \
  --batch-size 32 \
  --model efficientnet-b0
```

**Output**:
- Trained model saved to `weights/deepfake_model.pth`
- Training logs showing loss/accuracy

---

## 🐳 Docker Directory (`docker/`)

```
docker/
├── Dockerfile.backend        # Backend container image
│   ├── Base: python:3.11     # Python base image
│   ├── Install dependencies
│   ├── Copy code
│   └── Run uvicorn
│
├── Dockerfile.frontend       # Frontend container image
│   ├── Build stage           # Node 18 build
│   ├── Production stage      # Nginx-like server
│   └── Serve static
│
└── docker-compose.yml        # Orchestration
    ├── Backend service       # Port 8000
    ├── Frontend service      # Port 3000
    └── Volumes for persistence
```

### Docker Usage

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up

# Build from scratch
docker-compose -f docker/docker-compose.yml up --build

# Stop services
docker-compose down

# View logs
docker-compose logs -f backend
```

---

## 📚 Documentation Directory (`docs/`)

```
docs/
├── SETUP.md                  # Installation & setup instructions
│   ├── Prerequisites
│   ├── Local setup (Python/Node)
│   ├── Docker setup
│   ├── Environment configuration
│   └── Troubleshooting
│
├── ARCHITECTURE.md           # System design & decisions
│   ├── Architecture overview
│   ├── Technology choices
│   ├── Design trade-offs
│   ├── Performance bottlenecks
│   └── Future improvements
│
├── API.md                    # API reference
│   ├── Endpoints documentation
│   ├── Request/response formats
│   ├── Status codes
│   ├── Error handling
│   └── Code examples
│
├── TRAINING.md               # Model training guide
│   ├── Dataset preparation
│   ├── Training script
│   ├── Hyperparameters
│   ├── Advanced techniques
│   └── Evaluation metrics
│
└── DEPLOYMENT.md             # Production deployment
    ├── Docker deployment
    ├── AWS EC2 setup
    ├── Railway/Render setup
    ├── Nginx configuration
    ├── SSL/TLS setup
    ├── Monitoring & logging
    ├── Performance tuning
    └── Backup & recovery
```

---

## ⚙️ Configuration Files

**`.env.example`** (Backend)
- Template for environment variables
- Copy to `.env` and customize
- Variables: DEBUG, PORT, MODEL_PATH, DATABASE_URL, etc.

**`.env`** (Frontend)
- Frontend environment variables
- REACT_APP_API_URL: Points to backend API

**`.gitignore`**
- Excludes files from git
- Ignores: `__pycache__/`, `node_modules/`, `.env`, `weights/`, etc.

**`package.json`** (Frontend)
- Node.js project metadata
- Dependencies: react, axios, react-icons
- Scripts: start, build, test

**`requirements.txt`** (Backend)
- Python packages
- Core: fastapi, torch, torchvision
- Plus: pillow, opencv, sqlalchemy, pydantic

---

## 📊 Weights Directory (`weights/`)

```
weights/
└── deepfake_model.pth        # Trained model weights
                              # - Download from provided source
                              # - Or train using train.py
                              # - ~100MB file
                              # - Binary PyTorch format
```

**How to get weights**:
1. Download pretrained: (provide link)
2. Or train your own: `python models/training/train.py --data-dir data`
3. Place in `weights/` directory
4. Specify in `.env`: `MODEL_PATH=weights/deepfake_model.pth`

---

## 🗂️ Understanding Data Flow

### Prediction Flow

```
User Upload Image (Frontend)
    ↓
<form data multipart>
    ↓
Backend POST /predict
    ↓
File Validation
    ↓
Image Processing (PIL + PyTorch)
    ↓
Model Inference (Forward Pass)
    ↓
Grad-CAM Visualization
    ↓
Database Logging (Async)
    ↓
JSON Response
    ↓
Frontend Display Results
```

### Component Lifecycle

```
App.js
├─ User selects image
├─ ImageUploader displays preview
├─ User clicks "Analyze"
├─ App sends HTTP POST request
├─ Backend processes image
├─ Results received
└─ Results component displays prediction
```

---

## 🔧 Common Customizations

### Change Model

Edit `backend/app/config.py`:
```python
MODEL_NAME: str = "resnet50"  # Instead of efficientnet-b0
```

### Change UI Colors

Edit `frontend/src/index.css`:
```css
:root {
  --primary-color: #your-color;
}
```

### Add Custom Middleware

Edit `backend/main.py`:
```python
app.add_middleware(YourMiddleware)
```

### Add Database Fields

Edit `backend/app/utils/database.py`:
```python
class Prediction(Base):
    # Add new columns
    user_id = Column(String)
```

---

## 📈 Scale & Performance

### Single Machine Performance
- **Throughput**: 20-30 predictions/second (GPU)
- **Memory**: 2-4GB RAM + GPU VRAM
- **Disk**: 5GB (code + model + DB)

### For High Scale
- Use Kubernetes instead of Docker
- Add load balancer (Nginx/HAProxy)
- Database sharding (multiple PostgreSQL)
- Model serving (TensorFlow Serving/TorchServe)
- CDN for static assets

---

## 🎯 File Importance Ranking

**Critical** (App won't work without):
- `backend/main.py`
- `backend/app/api/routes.py`
- `backend/app/models/detector.py`
- `frontend/src/App.js`
- `docker/docker-compose.yml`

**Important** (Core functionality):
- `backend/app/utils/image_processing.py`
- `backend/app/utils/database.py`
- `frontend/src/components/*`
- `weights/deepfake_model.pth`

**Nice to Have** (Enhancement):
- `backend/app/utils/grad_cam.py`
- `frontend/src/index.css`
- Documentation files

---

## 📞 File Dependencies

```
main.py
  ├─ config.py (read settings)
  ├─ api/routes.py (include routes)
  └─ utils/database.py (initialize database)

routes.py
  ├─ config.py (file size limits, paths)
  ├─ models/detector.py (model inference)
  ├─ utils/image_processing.py (image preprocessing)
  ├─ utils/database.py (save predictions)
  └─ utils/grad_cam.py (visualization)

App.js (Frontend)
  └─ axios (make HTTP requests to /predict)
```

---

This structure is modular, scalable, and follows software engineering best practices.

For quick questions, check QUICKSTART.md
For detailed setup, check docs/SETUP.md

"""
API routes for deepfake detection
"""
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import numpy as np
from PIL import Image
import io
import torch

from app.config import settings
from app.utils.image_processing import preprocess_image
from app.utils.database import SessionLocal, Prediction, engine
from app.models.detector import DeepfakeDetector
from app.utils.grad_cam import generate_grad_cam

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize model
detector = None


def init_model():
    """Initialize the model on startup"""
    global detector
    if detector is None:
        try:
            detector = DeepfakeDetector(
                model_name=settings.MODEL_NAME,
                device=settings.DEVICE,
                model_path=settings.MODEL_PATH
            )
            logger.info(f"Model loaded successfully from {settings.MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


def save_prediction_to_db(
    filename: str,
    prediction: str,
    confidence: float,
    explanation: str,
    image_size: tuple
):
    """Save prediction to database"""
    try:
        db = SessionLocal()
        pred = Prediction(
            filename=filename,
            prediction=prediction,
            confidence=confidence,
            explanation=explanation,
            image_width=image_size[0],
            image_height=image_size[1]
        )
        db.add(pred)
        db.commit()
        db.close()
        logger.info(f"Saved prediction for {filename} to database")
    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")


@router.post("/predict")
async def predict(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Predict if uploaded image is deepfake or authentic
    
    Args:
        file: Image file (JPG or PNG)
        
    Returns:
        JSON response with prediction, confidence, and explanation
    """
    
    # Initialize model if not already done
    if detector is None:
        init_model()
    
    # Validate file type
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Validate file size
    try:
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=500, detail="Error reading file")
    
    # Process image
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_size = image.size
        
        # Preprocess
        tensor_image = preprocess_image(image)
        
        # Make prediction
        prediction, confidence, logits = detector.predict(tensor_image)
        
        # Generate Grad-CAM
        try:
            cam_array = generate_grad_cam(detector, tensor_image)
            explanation = "Grad-CAM heatmap generated"
        except Exception as e:
            logger.warning(f"Could not generate Grad-CAM: {e}")
            cam_array = None
            explanation = "Prediction made without visualization"
        
        # Prepare response
        response = {
            "prediction": prediction,
            "confidence": float(confidence),
            "explanation": explanation,
            "filename": file.filename,
            "model": settings.MODEL_NAME,
            "device": settings.DEVICE,
            "grad_cam_available": cam_array is not None
        }
        
        # Save to database in background
        background_tasks.add_task(
            save_prediction_to_db,
            filename=file.filename,
            prediction=prediction,
            confidence=float(confidence),
            explanation=explanation,
            image_size=image_size
        )
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "app": settings.APP_NAME, "version": settings.APP_VERSION}


@router.get("/model-info")
async def model_info():
    """Get model information"""
    if detector is None:
        init_model()
    
    return {
        "model_name": settings.MODEL_NAME,
        "device": settings.DEVICE,
        "model_path": settings.MODEL_PATH,
        "parameters": detector.get_model_info(),
    }

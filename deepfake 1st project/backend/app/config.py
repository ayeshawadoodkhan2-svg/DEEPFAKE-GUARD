"""
Application configuration settings
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # App
    APP_NAME: str = "Deepfake Detector API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # File upload
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: list = ["jpg", "jpeg", "png"]
    UPLOAD_DIR: str = "uploads"
    
    # Model
    MODEL_NAME: str = "efficientnet-b0"
    MODEL_PATH: str = "weights/deepfake_model.pth"
    DEVICE: str = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Database
    DATABASE_URL: str = "sqlite:///./predictions.db"
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8080", "*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"


settings = Settings()

"""
Database models and configuration
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=settings.DEBUG
)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class
Base = declarative_base()


class Prediction(Base):
    """Prediction database model"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    prediction = Column(String)  # "Real" or "Deepfake"
    confidence = Column(Float)  # 0-100
    explanation = Column(String, nullable=True)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, filename={self.filename}, prediction={self.prediction}, confidence={self.confidence})>"


# Create tables
def init_db():
    """Initialize database"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")


# Call on import
init_db()

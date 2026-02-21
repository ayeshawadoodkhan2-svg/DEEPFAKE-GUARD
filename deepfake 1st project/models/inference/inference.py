"""
Model inference script
"""
import torch
import argparse
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from app.models.detector import DeepfakeDetector
from app.utils.image_processing import preprocess_image
from PIL import Image


def predict_image(image_path: str, model_path: str, device: str = "cuda"):
    """
    Predict if image is deepfake or authentic
    
    Args:
        image_path: Path to image file
        model_path: Path to model weights
        device: Device to use
    """
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image_path}")
    print(f"Image size: {image.size}")
    
    # Initialize detector
    detector = DeepfakeDetector(
        model_name="efficientnet-b0",
        device=device,
        model_path=model_path
    )
    print(f"Model loaded from {model_path}")
    
    # Preprocess
    tensor_image = preprocess_image(image)
    
    # Predict
    prediction, confidence, logits = detector.predict(tensor_image)
    
    print(f"\n{'='*50}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"{'='*50}")
    
    return {"prediction": prediction, "confidence": confidence}


def main():
    parser = argparse.ArgumentParser(description="Run inference on image")
    parser.add_argument("image", type=str, help="Path to image file")
    parser.add_argument("--model", type=str, default="weights/deepfake_model.pth", help="Path to model")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    predict_image(args.image, args.model, args.device)


if __name__ == "__main__":
    main()

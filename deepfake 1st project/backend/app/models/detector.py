"""
Deepfake detection model
"""
import torch
import torch.nn as nn
import torchvision.models as models
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """
    Deepfake detection model using EfficientNet or ResNet
    """
    
    def __init__(self, model_name: str = "efficientnet-b0", device: str = "cpu", model_path: str = None):
        """
        Initialize the deepfake detector
        
        Args:
            model_name: Model architecture ("efficientnet-b0" or "resnet50")
            device: Device to run model on ("cpu" or "cuda")
            model_path: Path to pretrained weights
        """
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = self._build_model(model_name)
        
        if model_path and torch.cuda.is_available() or device == "cpu":
            try:
                self._load_weights(model_path)
                logger.info(f"Loaded weights from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load weights from {model_path}: {e}")
                logger.info("Using randomly initialized weights")
        
        self.model.to(self.device)
        self.model.eval()
    
    def _build_model(self, model_name: str) -> nn.Module:
        """Build the model architecture"""
        if "efficientnet" in model_name.lower():
            if model_name.lower() == "efficientnet-b0":
                model = models.efficientnet_b0(weights="DEFAULT")
            elif model_name.lower() == "efficientnet-b1":
                model = models.efficientnet_b1(weights="DEFAULT")
            elif model_name.lower() == "efficientnet-b2":
                model = models.efficientnet_b2(weights="DEFAULT")
            else:
                model = models.efficientnet_b0(weights="DEFAULT")
            
            # Modify classifier for binary classification
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 2)
            
        elif "resnet" in model_name.lower():
            if model_name.lower() == "resnet50":
                model = models.resnet50(weights="DEFAULT")
            elif model_name.lower() == "resnet101":
                model = models.resnet101(weights="DEFAULT")
            else:
                model = models.resnet50(weights="DEFAULT")
            
            # Modify final layer for binary classification
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
            
        else:
            # Default to EfficientNet-B0
            model = models.efficientnet_b0(weights="DEFAULT")
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 2)
        
        return model
    
    def _load_weights(self, model_path: str):
        """Load model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
    
    def predict(self, image_tensor: torch.Tensor) -> Tuple[str, float, torch.Tensor]:
        """
        Make prediction on image tensor
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Tuple of (prediction_label, confidence_score, logits)
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # Map to labels
            labels = ["Real", "Deepfake"]
            prediction = labels[predicted_class.item()]
            confidence_score = confidence.item() * 100
            
            return prediction, confidence_score, logits
    
    def get_model_info(self) -> dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
        }
    
    def get_layers(self):
        """Get model for Grad-CAM"""
        return self.model

"""
Grad-CAM visualization for model interpretability
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Layer to compute gradients for
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Store activations"""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Store gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class (if None, uses highest predicted class)
            
        Returns:
            CAM heatmap as numpy array
        """
        batch_size, channels, height, width = input_tensor.size()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        target = output[0, target_class]
        target.backward()
        
        # Compute Grad-CAM
        if self.gradients is None or self.activations is None:
            logger.warning("Could not extract gradients or activations")
            return None
        
        # Global average pooling of gradients
        pooled_gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weight activations
        cam = (pooled_gradients * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input size
        cam = F.interpolate(cam, size=(height, width), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        return cam


def generate_grad_cam(detector, image_tensor: torch.Tensor) -> Optional[np.ndarray]:
    """
    Generate Grad-CAM heatmap
    
    Args:
        detector: DeepfakeDetector instance
        image_tensor: Preprocessed image tensor
        
    Returns:
        Grad-CAM heatmap as numpy array or None if failed
    """
    try:
        model = detector.get_layers()
        
        # Get the last convolutional layer
        # This varies by architecture
        if hasattr(model, 'features'):  # EfficientNet
            target_layer = model.features[-1]
        elif hasattr(model, 'layer4'):  # ResNet
            target_layer = model.layer4[-1]
        else:
            logger.warning("Could not find target layer for Grad-CAM")
            return None
        
        # Create Grad-CAM
        grad_cam = GradCAM(model, target_layer)
        
        # Generate CAM
        cam = grad_cam.generate_cam(image_tensor)
        
        return cam
        
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {e}")
        return None


def apply_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Apply CAM heatmap to image
    
    Args:
        image: Original image (0-255, BGR)
        cam: CAM heatmap (0-1)
        alpha: Transparency of overlay
        
    Returns:
        Image with heatmap overlay
    """
    # Convert CAM to color heatmap
    cam_colored = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)
    
    return overlay

"""
Image preprocessing utilities
"""
import torch
import torchvision.transforms as transforms
import logging

logger = logging.getLogger(__name__)

# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Image size for EfficientNet / ResNet
IMAGE_SIZE = 224


def preprocess_image(image, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """
    Preprocess PIL image to tensor
    
    Args:
        image: PIL Image object
        image_size: Target image size (default 224 for standard CNNs)
        
    Returns:
        Preprocessed tensor
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    tensor = transform(image)
    return tensor


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize tensor back to original range [0, 1]
    
    Args:
        tensor: Normalized tensor
        
    Returns:
        Denormalized tensor
    """
    for t, m, s in zip(tensor, IMAGENET_MEAN, IMAGENET_STD):
        t.mul_(s).add_(m)
    return tensor


def get_preprocessing_transform():
    """Get preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

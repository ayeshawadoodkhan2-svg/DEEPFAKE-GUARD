"""
Model training script for deepfake detection
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DeepfakeDataset(Dataset):
    """
    Custom dataset for deepfake detection
    Expected directory structure:
    - data/
        - real/
            - image1.jpg
            - image2.jpg
        - deepfake/
            - image1.jpg
            - image2.jpg
    """
    
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load real images (label 0)
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for img in real_dir.glob("*"):
                if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.images.append(str(img))
                    self.labels.append(0)
        
        # Load deepfake images (label 1)
        deepfake_dir = self.data_dir / "deepfake"
        if deepfake_dir.exists():
            for img in deepfake_dir.glob("*"):
                if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.images.append(str(img))
                    self.labels.append(1)
        
        logger.info(f"Loaded {len(self.images)} images")
        logger.info(f"Real: {self.labels.count(0)}, Deepfake: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        image_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DeepfakeDetectorTrainer:
    """Train deepfake detection model"""
    
    def __init__(self, model_name="efficientnet-b0", device="cuda", learning_rate=0.001):
        self.device = torch.device(device)
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {self.device}")
    
    def _build_model(self):
        """Build model"""
        if "efficientnet" in self.model_name.lower():
            if self.model_name == "efficientnet-b0":
                model = models.efficientnet_b0(weights="DEFAULT")
            else:
                model = models.efficientnet_b0(weights="DEFAULT")
            
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 2)
        
        else:  # ResNet
            if self.model_name == "resnet50":
                model = models.resnet50(weights="DEFAULT")
            else:
                model = models.resnet50(weights="DEFAULT")
            
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
        
        return model
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({
                "loss": loss.item(),
                "accuracy": 100 * correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader=None, epochs=10, save_path="weights/deepfake_model.pth"):
        """Train the model"""
        best_accuracy = 0
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            if val_loader:
                val_loss, val_acc = self.validate(val_loader)
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save best model
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    self.save_model(save_path)
                    logger.info(f"Saved best model with accuracy: {val_acc:.2f}%")
            else:
                # No validation set, save every epoch
                self.save_model(save_path)
    
    def save_model(self, path):
        """Save model"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to dataset")
    parser.add_argument("--model", type=str, default="efficientnet-b0", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default="weights/deepfake_model.pth", help="Output path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load dataset
    dataset = DeepfakeDataset(args.data_dir, transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Train
    trainer = DeepfakeDetectorTrainer(
        model_name=args.model,
        device=args.device,
        learning_rate=args.learning_rate
    )
    
    trainer.train(train_loader, val_loader, epochs=args.epochs, save_path=args.output)


if __name__ == "__main__":
    main()

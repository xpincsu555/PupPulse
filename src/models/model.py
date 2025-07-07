"""
Model architecture definitions for PupPulse emotion classification.
Supports both custom CNN and transfer learning approaches.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict
import logging

class CustomCNN(nn.Module):
    """
    Custom CNN architecture for dog emotion classification.
    Simple but effective architecture with batch normalization and dropout.
    """
    
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.5):
        """
        Initialize custom CNN.
        
        Args:
            num_classes (int): Number of emotion classes
            dropout_rate (float): Dropout probability
        """
        super(CustomCNN, self).__init__()
        
        # Convolutional layers with batch normalization
        self.features = nn.Sequential(
            # First block: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224x224 -> 112x112
            
            # Second block: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112 -> 56x56
            
            # Third block: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56 -> 28x28
            
            # Fourth block: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            
            # Fifth block: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))  # Adaptive pooling to 7x7
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.classifier(x)
        return x

class TransferLearningModel(nn.Module):
    """
    Transfer learning model using pre-trained ResNet architectures.
    Fine-tunes the final classification layer for dog emotion classification.
    """
    
    def __init__(self, model_name: str = 'resnet18', num_classes: int = 4, 
                 pretrained: bool = True, dropout_rate: float = 0.5):
        """
        Initialize transfer learning model.
        
        Args:
            model_name (str): Name of the base model (resnet18, resnet34, resnet50)
            num_classes (int): Number of emotion classes
            pretrained (bool): Whether to use ImageNet pretrained weights
            dropout_rate (float): Dropout probability
        """
        super(TransferLearningModel, self).__init__()
        
        self.model_name = model_name
        
        # Load pre-trained model
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        logging.info(f"Initialized {model_name} with {num_classes} classes")
        if pretrained:
            logging.info("Using ImageNet pretrained weights")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output logits
        """
        return self.backbone(x)

def create_model(config: Dict) -> nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        nn.Module: Initialized model
    """
    model_config = config['model']
    model_name = model_config['name']
    num_classes = model_config['num_classes']
    dropout_rate = model_config['dropout_rate']
    
    if model_name == 'custom_cnn':
        model = CustomCNN(num_classes=num_classes, dropout_rate=dropout_rate)
        logging.info("Created custom CNN model")
    elif model_name in ['resnet18', 'resnet34', 'resnet50']:
        pretrained = model_config['pretrained']
        model = TransferLearningModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
        logging.info(f"Created {model_name} transfer learning model")
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def get_device(config: Dict) -> torch.device:
    """
    Get the appropriate device for training/inference.
    Supports M1 Mac MPS, CUDA, and CPU fallback.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        torch.device: Device to use for computation
    """
    device_config = config.get('device', {})
    use_mps = device_config.get('use_mps', True)
    fallback_to_cpu = device_config.get('fallback_to_cpu', True)
    
    # Check for MPS (Metal Performance Shaders) on M1 Mac
    if use_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.info("Using Metal Performance Shaders (MPS) for M1 Mac")
    # Check for CUDA
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("Using CUDA GPU")
    # Fallback to CPU
    elif fallback_to_cpu:
        device = torch.device('cpu')
        logging.info("Using CPU (no GPU acceleration available)")
    else:
        raise RuntimeError("No suitable device found and CPU fallback disabled")
    
    return device

def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

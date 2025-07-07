"""
Data loading and preprocessing utilities for PupPulse emotion classification.
Handles dataset creation, train/validation splits, and data transformations.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import yaml
from typing import Tuple, Dict, List
import logging

class DogEmotionDataset(Dataset):
    """
    Custom dataset class for dog emotion classification.
    Loads images from organized directories (angry/, happy/, relaxed/, sad/).
    """
    
    def __init__(self, data_root: str, transform=None):
        """
        Initialize dataset.
        
        Args:
            data_root (str): Path to directory containing emotion subdirectories
            transform: Torchvision transforms to apply to images
        """
        self.data_root = data_root
        self.transform = transform
        
        # Define emotion classes and their indices
        self.classes = ['angry', 'happy', 'relaxed', 'sad']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = self._load_samples()
        
        logging.info(f"Loaded {len(self.samples)} images from {len(self.classes)} emotion classes")
        self._log_class_distribution()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their corresponding labels."""
        samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_root, class_name)
            
            if not os.path.exists(class_dir):
                logging.warning(f"Directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files in the class directory
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, filename)
                    samples.append((img_path, class_idx))
        
        return samples
    
    def _log_class_distribution(self):
        """Log the distribution of samples across classes."""
        class_counts = {cls: 0 for cls in self.classes}
        
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] += 1
        
        logging.info("Class distribution:")
        for cls, count in class_counts.items():
            logging.info(f"  {cls}: {count} images")
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]
        
        try:
            # Load and convert image to RGB
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms if specified
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            blank_image = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, label

def get_transforms(config: Dict) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create training and validation transforms based on configuration.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    input_size = config['data']['input_size']
    aug_config = config['data']['augmentation']
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(aug_config['rotation_degrees']),
        transforms.RandomHorizontalFlip(p=aug_config['horizontal_flip_prob']),
        transforms.ColorJitter(
            brightness=aug_config['brightness'],
            contrast=aug_config['contrast']
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug_config['normalize_mean'],
            std=aug_config['normalize_std']
        )
    ])
    
    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug_config['normalize_mean'],
            std=aug_config['normalize_std']
        )
    ])
    
    return train_transform, val_transform

def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_root = config['paths']['data_root']
    batch_size = config['training']['batch_size']
    train_split = config['data']['train_split']
    num_workers = config['data']['num_workers']
    random_seed = config['data']['random_seed']
    
    # Get transforms
    train_transform, val_transform = get_transforms(config)
    
    # Create full dataset with training transforms first to get split indices
    full_dataset = DogEmotionDataset(data_root, transform=None)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    logging.info(f"Dataset split - Train: {train_size}, Validation: {val_size}")
    
    # Create train/validation split
    generator = torch.Generator().manual_seed(random_seed)
    train_indices, val_indices = random_split(
        range(total_size), 
        [train_size, val_size], 
        generator=generator
    )
    
    # Create separate datasets with appropriate transforms
    train_dataset = DogEmotionDataset(data_root, transform=train_transform)
    val_dataset = DogEmotionDataset(data_root, transform=val_transform)
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    logging.info(f"Loaded configuration from {config_path}")
    return config

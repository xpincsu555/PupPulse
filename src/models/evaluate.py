"""
Evaluation script for PupPulse dog emotion classification model.
Computes detailed metrics including confusion matrix and per-class accuracy.
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import json
from typing import Dict, List, Tuple

# Import custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loader import create_data_loaders, load_config
from models.model import create_model, get_device

def load_model_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    """
    Load model weights from checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded model weights from {checkpoint_path}")
    
    # Log checkpoint information
    if 'epoch' in checkpoint:
        logging.info(f"Checkpoint epoch: {checkpoint['epoch'] + 1}")
    if 'val_acc' in checkpoint:
        logging.info(f"Checkpoint validation accuracy: {checkpoint['val_acc']:.2f}%")

def evaluate_model(model: nn.Module, data_loader, device: torch.device, 
                  class_names: List[str]) -> Dict:
    """
    Evaluate model on given dataset.
    
    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        class_names: List of class names
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    correct_predictions = 0
    total_samples = 0
    
    logging.info("Running model evaluation...")
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Collect predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update counters
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    # Calculate overall accuracy
    accuracy = 100.0 * correct_predictions / total_samples
    
    # Calculate detailed metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, labels=range(len(class_names))
    )
    
    # Calculate macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    
    # Create evaluation results dictionary
    results = {
        'accuracy': accuracy,
        'precision_macro': precision_macro * 100,
        'recall_macro': recall_macro * 100,
        'f1_macro': f1_macro * 100,
        'precision_weighted': precision_weighted * 100,
        'recall_weighted': recall_weighted * 100,
        'f1_weighted': f1_weighted * 100,
        'per_class_metrics': {}
    }
    
    # Per-class metrics
    for i, class_name in enumerate(class_names):
        results['per_class_metrics'][class_name] = {
            'precision': precision[i] * 100,
            'recall': recall[i] * 100,
            'f1_score': f1[i] * 100,
            'support': int(support[i])
        }
    
    return results, all_predictions, all_labels

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                         class_names: List[str], output_dir: str):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot confusion matrix with counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Plot confusion matrix with percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Percentages)')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Confusion matrix saved to {save_path}")

def plot_class_distribution(y_true: List[int], class_names: List[str], 
                           output_dir: str):
    """
    Plot class distribution in the evaluation dataset.
    
    Args:
        y_true: True labels
        class_names: List of class names
        output_dir: Directory to save plot
    """
    # Count samples per class
    unique, counts = np.unique(y_true, return_counts=True)
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar([class_names[i] for i in unique], counts, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title('Class Distribution in Evaluation Dataset')
    plt.xlabel('Emotion Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Save plot
    save_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Class distribution plot saved to {save_path}")

def save_evaluation_report(results: Dict, output_dir: str):
    """
    Save detailed evaluation report to JSON and text files.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save reports
    """
    # Save JSON report
    json_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save text report
    text_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(text_path, 'w') as f:
        f.write("PupPulse Dog Emotion Classification - Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"Macro Precision: {results['precision_macro']:.2f}%\n")
        f.write(f"Macro Recall: {results['recall_macro']:.2f}%\n")
        f.write(f"Macro F1-Score: {results['f1_macro']:.2f}%\n")
        f.write(f"Weighted Precision: {results['precision_weighted']:.2f}%\n")
        f.write(f"Weighted Recall: {results['recall_weighted']:.2f}%\n")
        f.write(f"Weighted F1-Score: {results['f1_weighted']:.2f}%\n\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
        f.write("-" * 60 + "\n")
        
        for class_name, metrics in results['per_class_metrics'].items():
            f.write(f"{class_name:<10} {metrics['precision']:<12.2f} "
                   f"{metrics['recall']:<10.2f} {metrics['f1_score']:<10.2f} "
                   f"{metrics['support']:<10}\n")
    
    logging.info(f"Evaluation report saved to {json_path} and {text_path}")

def main():
    """Main function to parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate PupPulse emotion classification model')
    parser.add_argument('--config', type=str, 
                       default='configs/emotion_classification.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights file')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting PupPulse model evaluation")
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    device = get_device(config)
    
    # Create data loaders (using validation split for evaluation)
    _, val_loader = create_data_loaders(config)
    
    # Define class names
    class_names = ['angry', 'happy', 'relaxed', 'sad']
    
    # Create and load model
    model = create_model(config)
    model = model.to(device)
    load_model_checkpoint(model, args.weights, device)
    
    # Run evaluation
    results, predictions, true_labels = evaluate_model(
        model, val_loader, device, class_names
    )
    
    # Print results
    print("\nEVALUATION RESULTS:")
    print("=" * 50)
    print(f"Overall Accuracy: {results['accuracy']:.2f}%")
    print(f"Macro F1-Score: {results['f1_macro']:.2f}%")
    print(f"Weighted F1-Score: {results['f1_weighted']:.2f}%")
    
    print("\nPer-Class Results:")
    print("-" * 50)
    for class_name, metrics in results['per_class_metrics'].items():
        print(f"{class_name.capitalize():<8}: Precision={metrics['precision']:.1f}%, "
              f"Recall={metrics['recall']:.1f}%, F1={metrics['f1_score']:.1f}%, "
              f"Support={metrics['support']}")
    
    # Generate and save visualizations
    plot_confusion_matrix(true_labels, predictions, class_names, args.output_dir)
    plot_class_distribution(true_labels, class_names, args.output_dir)
    
    # Save detailed report
    save_evaluation_report(results, args.output_dir)
    
    logging.info(f"Evaluation completed. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()

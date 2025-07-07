"""
Single image prediction script for PupPulse dog emotion classification.
Loads trained model and predicts emotion for a given dog image.
"""

import os
import argparse
import logging
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import json

# Import custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loader import load_config
from models.model import create_model, get_device

class DogEmotionPredictor:
    """
    Class for predicting dog emotions from single images.
    """
    
    def __init__(self, config_path: str, weights_path: str):
        """
        Initialize the predictor.
        
        Args:
            config_path (str): Path to configuration file
            weights_path (str): Path to model weights
        """
        self.config = load_config(config_path)
        self.device = get_device(self.config)
        self.class_names = ['angry', 'happy', 'relaxed', 'sad']
        
        # Load model
        self.model = create_model(self.config)
        self.model = self.model.to(self.device)
        self._load_weights(weights_path)
        
        # Create transform
        self.transform = self._create_transform()
        
        logging.info("PupPulse emotion predictor initialized")
    
    def _load_weights(self, weights_path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logging.info(f"Loaded model weights from {weights_path}")
    
    def _create_transform(self):
        """Create image preprocessing transform."""
        input_size = self.config['data']['input_size']
        aug_config = self.config['data']['augmentation']
        
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug_config['normalize_mean'],
                std=aug_config['normalize_std']
            )
        ])
        
        return transform
    
    def predict(self, image_path: str, return_probabilities: bool = False):
        """
        Predict emotion for a single image.
        
        Args:
            image_path (str): Path to input image
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Prepare results
            results = {
                'predicted_emotion': self.class_names[predicted_class],
                'confidence': confidence * 100,
                'image_path': image_path
            }
            
            if return_probabilities:
                class_probabilities = {}
                for i, class_name in enumerate(self.class_names):
                    class_probabilities[class_name] = probabilities[0][i].item() * 100
                results['class_probabilities'] = class_probabilities
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            return None
    
    def predict_batch(self, image_paths: list, return_probabilities: bool = False):
        """
        Predict emotions for multiple images.
        
        Args:
            image_paths (list): List of image paths
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            result = self.predict(image_path, return_probabilities)
            if result:
                results.append(result)
        
        return results

def display_prediction_results(results: dict):
    """
    Display prediction results in a formatted way.
    
    Args:
        results (dict): Prediction results dictionary
    """
    print("\n" + "=" * 60)
    print("🐕 PUPPULSE EMOTION PREDICTION RESULTS 🐕")
    print("=" * 60)
    
    print(f"Image: {os.path.basename(results['image_path'])}")
    print(f"Predicted Emotion: {results['predicted_emotion'].upper()}")
    print(f"Confidence: {results['confidence']:.1f}%")
    
    if 'class_probabilities' in results:
        print("\nClass Probabilities:")
        print("-" * 30)
        for emotion, prob in results['class_probabilities'].items():
            # Create a simple text-based progress bar
            bar_length = 20
            filled_length = int(bar_length * prob / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"{emotion.capitalize():<8}: {bar} {prob:5.1f}%")
    
    print("=" * 60)

def validate_image_path(image_path: str) -> bool:
    """
    Validate if the image path exists and is a valid image file.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    file_extension = os.path.splitext(image_path)[1].lower()
    
    if file_extension not in valid_extensions:
        print(f"Error: Unsupported image format: {file_extension}")
        print(f"Supported formats: {', '.join(valid_extensions)}")
        return False
    
    return True

def main():
    """Main function to parse arguments and run prediction."""
    parser = argparse.ArgumentParser(
        description='Predict dog emotion from image using PupPulse model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/inference/predict.py --image data/raw/images/happy/dog1.jpg
  python src/inference/predict.py --image photo.jpg --probabilities --save-results
  python src/inference/predict.py --image dog.png --config configs/custom_config.yaml
        """
    )
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input dog image')
    parser.add_argument('--config', type=str, 
                       default='configs/emotion_classification.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--weights', type=str, 
                       default='outputs/best_model.pth',
                       help='Path to trained model weights')
    parser.add_argument('--probabilities', action='store_true',
                       help='Show probabilities for all classes')
    parser.add_argument('--save-results', action='store_true',
                       help='Save prediction results to JSON file')
    parser.add_argument('--output-file', type=str,
                       help='Output file path for saved results')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate inputs
    if not validate_image_path(args.image):
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return
    
    if not os.path.exists(args.weights):
        print(f"Error: Model weights file not found: {args.weights}")
        return
    
    try:
        # Initialize predictor
        print("Loading PupPulse emotion classification model...")
        predictor = DogEmotionPredictor(args.config, args.weights)
        
        # Make prediction
        print("Analyzing dog emotion...")
        results = predictor.predict(args.image, return_probabilities=args.probabilities)
        
        if results:
            # Display results
            display_prediction_results(results)
            
            # Save results if requested
            if args.save_results:
                if args.output_file:
                    output_path = args.output_file
                else:
                    # Generate default output filename
                    base_name = os.path.splitext(os.path.basename(args.image))[0]
                    output_path = f"{base_name}_emotion_prediction.json"
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"\nResults saved to: {output_path}")
        else:
            print("Failed to process the image. Please check the image file and try again.")
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        logging.error(f"Prediction failed: {e}", exc_info=True)

if __name__ == '__main__':
    main()

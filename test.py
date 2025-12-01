#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Image Caption Generator with Weather Classification
Tests the system on images and generates accuracy analysis with score curves
"""
import torch
import clip
import os
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess
import sys
import glob

# Weather classification setup (same as app.py)
WEATHER_PROMPTS = [
    "a photo of sunny weather with clear blue sky",
    "a photo of rainy weather with rain and clouds",
    "a photo of snowy weather with snow and winter scene",
    "a photo of cloudy weather with clouds in the sky",
    "a photo of foggy weather with fog and mist",
    "a photo of stormy weather with storm and lightning",
    "a photo of overcast weather with gray sky",
    "a photo of clear weather with bright sky"
]

WEATHER_LABELS = [
    "Sunny",
    "Rainy", 
    "Snowy",
    "Cloudy",
    "Foggy",
    "Stormy",
    "Overcast",
    "Clear"
]

# Global CLIP model (lazy loading)
_clip_model = None
_clip_preprocess = None
_device = None

def get_clip_model(clip_model='RN50x4', use_cpu=False):
    """Lazy load CLIP model with automatic GPU detection"""
    global _clip_model, _clip_preprocess, _device
    
    if _clip_model is None:
        # Auto-detect GPU: use GPU if available and use_cpu is False
        if use_cpu:
            _device = torch.device("cpu")
            print(f"Using CPU (forced by use_cpu=True)")
        elif torch.cuda.is_available():
            _device = torch.device("cuda")
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            _device = torch.device("cpu")
            print(f"GPU not available, using CPU")
        
        print(f"Loading CLIP model: {clip_model} on {_device}")
        _clip_model, _clip_preprocess = clip.load(clip_model, device=_device, jit=False)
        _clip_model.eval()
        print("CLIP model loaded successfully")
    
    return _clip_model, _clip_preprocess, _device

def classify_weather(image_path, clip_model='RN50x4', use_cpu=False):
    """
    Classify weather in image using CLIP Zero-shot classification
    
    Args:
        image_path: Path to the image file
        clip_model: CLIP model type
        use_cpu: Force CPU usage (default: False, auto-detect GPU)
        
    Returns:
        dict: Weather classification results with probabilities
    """
    try:
        model, preprocess, device = get_clip_model(clip_model, use_cpu)
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Prepare text prompts
        text_inputs = clip.tokenize(WEATHER_PROMPTS).to(device)
        
        # Get features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity (cosine similarity)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Convert to probabilities
            probabilities = similarity[0].cpu().numpy()
            
            # Get top prediction
            top_idx = np.argmax(probabilities)
            top_weather = WEATHER_LABELS[top_idx]
            top_probability = float(probabilities[top_idx])
            
            # Create results dictionary
            weather_scores = {
                label: float(prob) for label, prob in zip(WEATHER_LABELS, probabilities)
            }
            
            # Sort by probability
            sorted_scores = dict(sorted(weather_scores.items(), key=lambda x: x[1], reverse=True))
            
            return {
                "weather": top_weather,
                "confidence": top_probability,
                "all_scores": sorted_scores,
                "success": True
            }
            
    except Exception as e:
        print(f"Error in weather classification: {str(e)}")
        return {
            "weather": "Unknown",
            "confidence": 0.0,
            "all_scores": {},
            "success": False,
            "error": str(e)
        }

def enhance_caption_with_weather(caption, weather_result):
    """
    Enhance caption with weather information naturally
    
    Args:
        caption: Original caption text
        weather_result: Weather classification result
        
    Returns:
        str: Enhanced caption with weather information
    """
    if not weather_result or not weather_result.get('success'):
        return caption
    
    weather = weather_result.get('weather', '').lower()
    confidence = weather_result.get('confidence', 0.0)
    
    # Only enhance if confidence is high enough (>0.3)
    if confidence < 0.3:
        return caption
    
    # Clean up caption (remove special tokens if any)
    caption = caption.strip()
    if caption.startswith('<|endoftext|>'):
        caption = caption.replace('<|endoftext|>', '').strip()
    if caption.endswith('<|endoftext|>'):
        caption = caption.replace('<|endoftext|>', '').strip()
    
    # Remove trailing period for processing
    caption_clean = caption.rstrip('. ')
    caption_lower = caption_clean.lower()
    
    # Weather phrases that can be inserted naturally
    weather_phrases = {
        'sunny': 'on a sunny day',
        'rainy': 'in the rain',
        'snowy': 'in the snow',
        'cloudy': 'under cloudy skies',
        'foggy': 'in foggy conditions',
        'stormy': 'during a storm',
        'overcast': 'under overcast skies',
        'clear': 'on a clear day'
    }
    
    weather_phrase = weather_phrases.get(weather, '')
    
    if not weather_phrase:
        return caption
    
    # Check if caption already contains weather-related keywords
    weather_keywords = {
        'sunny': ['sunny', 'sun', 'bright', 'sunshine'],
        'rainy': ['rain', 'rainy', 'raining', 'wet', 'umbrella'],
        'snowy': ['snow', 'snowy', 'snowing', 'winter', 'white'],
        'cloudy': ['cloud', 'cloudy', 'clouds'],
        'foggy': ['fog', 'foggy', 'mist', 'haze'],
        'stormy': ['storm', 'stormy', 'lightning', 'thunder'],
        'overcast': ['overcast', 'gray', 'grey', 'gloomy'],
        'clear': ['clear', 'blue sky', 'bright']
    }
    
    # Check if weather is already mentioned
    keywords_to_check = weather_keywords.get(weather, [])
    if any(keyword in caption_lower for keyword in keywords_to_check):
        # Weather already mentioned, return original
        return caption
    
    # Try to insert weather phrase naturally
    if caption_clean:
        # Check if caption ends with common prepositions that we can extend
        if any(caption_lower.endswith(prep) for prep in ['in', 'on', 'at', 'with', 'under', 'during']):
            # Add weather after preposition
            caption_enhanced = caption_clean + ' ' + weather_phrase + '.'
        else:
            # Add weather at the end
            caption_enhanced = caption_clean + ' ' + weather_phrase + '.'
    else:
        caption_enhanced = caption
    
    return caption_enhanced

def generate_caption_for_image(image_path, model_weights, clip_model, use_cpu=False):
    """
    Generate caption for a single image using the prediction script
    
    Args:
        image_path: Path to the image file
        model_weights: Path to model weights
        clip_model: CLIP model type
        use_cpu: Force CPU usage (default: False, auto-detect GPU)
        
    Returns:
        str: Generated caption
    """
    try:
        # Create temporary directory for single image
        import tempfile
        import shutil
        import uuid
        
        temp_dir = tempfile.mkdtemp()
        # Get file extension and ensure lowercase
        file_ext = image_path.rsplit('.', 1)[1].lower() if '.' in image_path else 'jpg'
        unique_filename = f"{uuid.uuid4().hex}.{file_ext}"
        temp_image_path = os.path.join(temp_dir, unique_filename)
        
        # Copy image to temp directory
        shutil.copy2(image_path, temp_image_path)
        
        # Ensure the copied file has the correct extension that 03_Predict.py expects
        # 03_Predict.py searches for *.jpg, *.jpeg, *.png (lowercase)
        if file_ext not in ['jpg', 'jpeg', 'png']:
            # Convert to jpg if needed
            new_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.jpg")
            img = Image.open(temp_image_path).convert("RGB")
            img.save(new_path, "JPEG")
            os.remove(temp_image_path)
            temp_image_path = new_path
            unique_filename = os.path.basename(new_path)
        
        # Run prediction script
        output_file = os.path.join(temp_dir, 'captions.json')
        cmd = [
            sys.executable, '03_Predict.py',
            '--img_dir', temp_dir,
            '--weights', model_weights,
            '--clip_model', clip_model,
            '--output_file', output_file
        ]
        
        # Only add --use_cpu if explicitly requested
        if use_cpu:
            cmd.append('--use_cpu')
        
        # Run prediction
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Read caption
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                captions = json.load(f)
            caption = captions.get(unique_filename, "Caption not generated")
        else:
            caption = "Caption not generated"
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return caption
        
    except Exception as e:
        print(f"Error generating caption for {image_path}: {str(e)}")
        return "Error generating caption"


def test_images(image_dir, model_weights, clip_model, use_cpu=False, 
                ground_truth_file=None, output_dir='./test_results'):
    """
    Test images and generate captions with weather classification
    
    Args:
        image_dir: Directory containing test images
        model_weights: Path to model weights
        clip_model: CLIP model type
        use_cpu: Force CPU usage (default: False, auto-detect GPU)
        ground_truth_file: Optional JSON file with ground truth labels
        output_dir: Output directory for results
        
    Returns:
        dict: Test results with captions, weather, and accuracy metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check GPU availability
    if not use_cpu:
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print("Using GPU for processing")
        else:
            print("GPU not available, using CPU")
            use_cpu = True
    else:
        print("Using CPU (forced by use_cpu=True)")
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return None
    
    print(f"Found {len(image_files)} images to test")
    
    # Load ground truth if provided
    ground_truth = {}
    if ground_truth_file and os.path.exists(ground_truth_file):
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        print(f"Loaded ground truth from {ground_truth_file}")
    
    # Initialize CLIP model (for weather classification)
    print("\nLoading CLIP model for weather classification...")
    get_clip_model(clip_model, use_cpu)
    
    # Test results
    results = {
        'images': [],
        'weather_accuracy': 0.0,
        'caption_metrics': {},
        'weather_confidence_scores': [],
        'weather_distribution': {}
    }
    
    weather_correct = 0
    weather_total = 0
    confidence_scores = []
    weather_predictions = {}
    
    print("\nProcessing images...")
    for image_path in tqdm(image_files, desc="Testing images"):
        image_name = os.path.basename(image_path)
        
        try:
            # Generate caption
            print(f"\nProcessing: {image_name}")
            caption = generate_caption_for_image(image_path, model_weights, clip_model, use_cpu)
            
            # Classify weather
            weather_result = classify_weather(image_path, clip_model, use_cpu)
            
            # Enhance caption
            enhanced_caption = enhance_caption_with_weather(caption, weather_result)
            
            # Store result
            result_item = {
                'image': image_name,
                'caption': caption,
                'enhanced_caption': enhanced_caption,
                'weather': weather_result.get('weather', 'Unknown'),
                'confidence': weather_result.get('confidence', 0.0),
                'all_weather_scores': weather_result.get('all_scores', {})
            }
            
            # Check accuracy if ground truth provided
            if image_name in ground_truth:
                gt_weather = ground_truth[image_name].get('weather', '').lower()
                pred_weather = weather_result.get('weather', '').lower()
                
                if gt_weather:
                    weather_total += 1
                    if gt_weather == pred_weather:
                        weather_correct += 1
                        result_item['weather_correct'] = True
                    else:
                        result_item['weather_correct'] = False
                    result_item['ground_truth_weather'] = gt_weather
            
            results['images'].append(result_item)
            confidence_scores.append(weather_result.get('confidence', 0.0))
            
            # Update weather distribution
            weather = weather_result.get('weather', 'Unknown')
            weather_predictions[weather] = weather_predictions.get(weather, 0) + 1
            
            print(f"  Caption: {enhanced_caption}")
            print(f"  Weather: {weather} ({weather_result.get('confidence', 0.0)*100:.1f}%)")
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            results['images'].append({
                'image': image_name,
                'error': str(e)
            })
    
    # Calculate accuracy
    if weather_total > 0:
        results['weather_accuracy'] = weather_correct / weather_total
        print(f"\nWeather Classification Accuracy: {results['weather_accuracy']*100:.2f}% ({weather_correct}/{weather_total})")
    
    # Store metrics
    results['weather_confidence_scores'] = confidence_scores
    results['weather_distribution'] = weather_predictions
    results['average_confidence'] = np.mean(confidence_scores) if confidence_scores else 0.0
    results['std_confidence'] = np.std(confidence_scores) if confidence_scores else 0.0
    
    # Save results
    results_file = os.path.join(output_dir, 'test_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def plot_score_curves(results, output_dir='./test_results'):
    """
    Plot test score curves and analysis charts
    
    Args:
        results: Test results dictionary
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not results or not results.get('images'):
        print("No results to plot")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Confidence Score Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    confidence_scores = results.get('weather_confidence_scores', [])
    if confidence_scores:
        ax1.hist(confidence_scores, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidence_scores):.3f}')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Weather Classification Confidence Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Confidence Score Curve (per image)
    ax2 = fig.add_subplot(gs[0, 1])
    if confidence_scores:
        image_indices = range(len(confidence_scores))
        ax2.plot(image_indices, confidence_scores, marker='o', linestyle='-', 
                color='blue', markersize=6, label='Confidence Score')
        ax2.axhline(np.mean(confidence_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidence_scores):.3f}')
        ax2.set_xlabel('Image Index')
        ax2.set_ylabel('Confidence Score')
        ax2.set_title('Confidence Scores Across Test Images')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    
    # 3. Weather Distribution (Bar Chart)
    ax3 = fig.add_subplot(gs[1, 0])
    weather_dist = results.get('weather_distribution', {})
    if weather_dist:
        weathers = list(weather_dist.keys())
        counts = list(weather_dist.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(weathers)))
        ax3.bar(weathers, counts, color=colors, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Weather Type')
        ax3.set_ylabel('Count')
        ax3.set_title('Weather Type Distribution')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Accuracy Metrics (if ground truth available)
    ax4 = fig.add_subplot(gs[1, 1])
    accuracy = results.get('weather_accuracy', 0.0)
    if accuracy > 0:
        metrics = {
            'Accuracy': accuracy * 100,
            'Average Confidence': results.get('average_confidence', 0.0) * 100
        }
        bars = ax4.bar(metrics.keys(), metrics.values(), color=['green', 'blue'], 
                      edgecolor='black', alpha=0.7)
        ax4.set_ylabel('Percentage (%)')
        ax4.set_title('Weather Classification Metrics')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'No ground truth data\navailable for accuracy', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=12, style='italic')
        ax4.set_title('Accuracy Metrics (No Ground Truth)')
    
    # 5. Top Weather Scores (Box Plot)
    ax5 = fig.add_subplot(gs[2, 0])
    if results.get('images'):
        # Get top weather score for each image
        top_scores = []
        for item in results['images']:
            if 'all_weather_scores' in item and item['all_weather_scores']:
                top_score = list(item['all_weather_scores'].values())[0]
                top_scores.append(top_score)
        
        if top_scores:
            ax5.boxplot([top_scores], labels=['Top Weather Score'], 
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax5.set_ylabel('Score')
            ax5.set_title('Top Weather Score Distribution (Box Plot)')
            ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Cumulative Accuracy Curve (if ground truth available)
    ax6 = fig.add_subplot(gs[2, 1])
    if accuracy > 0 and results.get('images'):
        # Calculate cumulative accuracy
        correct_count = 0
        cumulative_accuracy = []
        for i, item in enumerate(results['images']):
            if 'weather_correct' in item:
                if item['weather_correct']:
                    correct_count += 1
                cumulative_accuracy.append(correct_count / (i + 1))
        
        if cumulative_accuracy:
            ax6.plot(range(len(cumulative_accuracy)), 
                    [acc * 100 for acc in cumulative_accuracy],
                    marker='o', linestyle='-', color='green', markersize=4)
            ax6.axhline(accuracy * 100, color='red', linestyle='--', 
                       label=f'Final Accuracy: {accuracy*100:.1f}%')
            ax6.set_xlabel('Number of Images Processed')
            ax6.set_ylabel('Cumulative Accuracy (%)')
            ax6.set_title('Cumulative Accuracy Curve')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim(0, 100)
    else:
        ax6.text(0.5, 0.5, 'No ground truth data\nfor accuracy curve', 
                ha='center', va='center', transform=ax6.transAxes,
                fontsize=12, style='italic')
        ax6.set_title('Cumulative Accuracy Curve (No Ground Truth)')
    
    # Save figure
    plt.suptitle('Test Results Analysis - Image Caption Generator with Weather Classification', 
                fontsize=16, fontweight='bold', y=0.995)
    output_file = os.path.join(output_dir, 'test_score_curves.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Score curves saved to: {output_file}")


def create_ground_truth_template(image_dir, output_file='ground_truth_template.json'):
    """
    Create a template JSON file for ground truth labels
    
    Args:
        image_dir: Directory containing images
        output_file: Output file path
    """
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    ground_truth = {}
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        ground_truth[image_name] = {
            'weather': '',  # Fill in: Sunny, Rainy, Snowy, Cloudy, Foggy, Stormy, Overcast, Clear
            'caption': ''   # Optional: reference caption
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    
    print(f"Ground truth template created: {output_file}")
    print("Please fill in the weather labels and captions manually.")


def main():
    parser = argparse.ArgumentParser(description='Test Image Caption Generator with Weather Classification')
    parser.add_argument('--image_dir', type=str, default='./Test_Set',
                       help='Directory containing test images (default: ./Test_Set)')
    parser.add_argument('--model_weights', type=str, default='./checkpoints/clip_pro_prefix-001.pt',
                       help='Model weights file path')
    parser.add_argument('--clip_model', type=str, default='RN50x4',
                       help='CLIP model type')
    parser.add_argument('--ground_truth', type=str, default=None,
                       help='JSON file with ground truth labels (optional)')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Output directory for results (default: ./test_results)')
    parser.add_argument('--use_cpu', action='store_true',
                       help='Force CPU usage (default: False, auto-detect and use GPU if available)')
    parser.add_argument('--create_template', action='store_true',
                       help='Create ground truth template file')
    
    args = parser.parse_args()
    
    # Create template if requested
    if args.create_template:
        create_ground_truth_template(args.image_dir, 'ground_truth.json')
        return
    
    # Check if image directory exists
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return
    
    # Check if model weights exist
    if not os.path.exists(args.model_weights):
        print(f"Warning: Model weights not found: {args.model_weights}")
        print("Please check the model weights path.")
    
    # Check GPU availability and display info
    device_info = "CPU (forced)" if args.use_cpu else ("GPU" if torch.cuda.is_available() else "CPU (no GPU available)")
    if torch.cuda.is_available() and not args.use_cpu:
        device_info += f" - {torch.cuda.get_device_name(0)}"
    
    # Run tests
    print("=" * 60)
    print("Image Caption Generator with Weather Classification - Test")
    print("=" * 60)
    print(f"Image Directory: {args.image_dir}")
    print(f"Model Weights: {args.model_weights}")
    print(f"CLIP Model: {args.clip_model}")
    print(f"Device: {device_info}")
    print(f"Ground Truth: {args.ground_truth if args.ground_truth else 'Not provided'}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)
    
    results = test_images(
        image_dir=args.image_dir,
        model_weights=args.model_weights,
        clip_model=args.clip_model,
        use_cpu=args.use_cpu,
        ground_truth_file=args.ground_truth,
        output_dir=args.output_dir
    )
    
    if results:
        # Plot score curves
        print("\nGenerating score curves...")
        plot_score_curves(results, output_dir=args.output_dir)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Total Images Tested: {len(results.get('images', []))}")
        if results.get('weather_accuracy', 0) > 0:
            print(f"Weather Accuracy: {results['weather_accuracy']*100:.2f}%")
        print(f"Average Confidence: {results.get('average_confidence', 0)*100:.2f}%")
        print(f"Std Confidence: {results.get('std_confidence', 0)*100:.2f}%")
        print(f"Weather Distribution: {results.get('weather_distribution', {})}")
        print("=" * 60)
    else:
        print("Test failed. Please check the error messages above.")


if __name__ == "__main__":
    main()


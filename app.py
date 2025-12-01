
"""
This is the main application file for the Image Caption Generator web app.
Extended with weather classification using CLIP Zero-shot.
"""
from flask import Flask, request, render_template, jsonify
import os
import json
import subprocess
import uuid
import shutil
import torch
import clip
from PIL import Image
import numpy as np

app = Flask(__name__)
 
# Configuration
UPLOAD_FOLDER = 'test_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CAPTION_OUTPUT = 'generated_captions.json'
MODEL_WEIGHTS = './checkpoints/clip_pro_prefix-001.pt'  
CLIP_MODEL = 'RN50x4'  
USE_CPU = True  
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Weather classification setup
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

def get_clip_model():
    """Lazy load CLIP model"""
    global _clip_model, _clip_preprocess, _device
    
    if _clip_model is None:
        _device = torch.device("cpu" if USE_CPU or not torch.cuda.is_available() else "cuda")
        print(f"Loading CLIP model: {CLIP_MODEL} on {_device}")
        _clip_model, _clip_preprocess = clip.load(CLIP_MODEL, device=_device, jit=False)
        _clip_model.eval()
        print("CLIP model loaded successfully")
    
    return _clip_model, _clip_preprocess, _device

def classify_weather(image_path):
    """
    Classify weather in image using CLIP Zero-shot classification
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Weather classification results with probabilities
    """
    try:
        model, preprocess, device = get_clip_model()
        
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
    
    # Try to insert weather naturally into the sentence
    # Look for common sentence patterns to insert weather
    
    # Pattern 1: "X stands/stands in front of Y" -> "X stands in front of Y on a sunny day"
    # Pattern 2: "X is Y" -> "X is Y on a sunny day"
    # Pattern 3: "There is/are X" -> "There is X on a sunny day"
    
    # Common verbs that indicate position/state
    position_verbs = ['stands', 'sits', 'lies', 'rests', 'is located', 'is situated']
    state_verbs = ['is', 'are', 'was', 'were']
    existence_verbs = ['there is', 'there are', 'there was', 'there were']
    
    # Check if we can insert weather before the end
    caption_words = caption_clean.split()
    
    # Try to insert weather phrase naturally
    # For most cases, append at the end with proper punctuation
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

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, generate caption and classify weather"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Clean up old files
        for f in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, f)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        # Save uploaded file
        unique_filename = f"{uuid.uuid4().hex}.{file.filename.rsplit('.', 1)[1].lower()}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        try:
            # Generate caption (existing functionality)
            import sys
            cmd = [
                sys.executable, '03_Predict.py',
                '--img_dir', UPLOAD_FOLDER,
                '--weights', MODEL_WEIGHTS,
                '--clip_model', CLIP_MODEL,
                '--output_file', CAPTION_OUTPUT
            ]
            
            if USE_CPU:
                cmd.append('--use_cpu')
                
            subprocess.run(cmd, check=True)
            
            # Read caption
            caption = "Caption not generated for this image."
            if os.path.exists(CAPTION_OUTPUT):
            with open(CAPTION_OUTPUT, 'r', encoding='utf-8') as f:
                captions = json.load(f)
            if unique_filename in captions:
                caption = captions[unique_filename]
            
            # Classify weather using CLIP Zero-shot
            weather_result = classify_weather(filepath)
            
            # Enhance caption with weather information
            enhanced_caption = enhance_caption_with_weather(caption, weather_result)
            
            # Prepare response
            response = {
                'success': True,
                'filename': unique_filename,
                'caption': enhanced_caption,
                'original_caption': caption,  # Keep original for reference
                'weather': weather_result
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the server is running"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
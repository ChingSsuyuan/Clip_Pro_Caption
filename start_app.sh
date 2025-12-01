#!/bin/bash
# Quick start script for the Image Caption Generator with Weather Classification

echo "=========================================="
echo "Image Caption Generator with Weather Classification"
echo "=========================================="
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source /Users/michaeljing/global_env/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    echo "Please check if the virtual environment exists at: /Users/michaeljing/global_env/bin/activate"
    exit 1
fi

echo "Virtual environment activated!"
echo ""

# Check if required packages are installed
echo "Checking required packages..."
python -c "import torch; import clip; import flask; import PIL" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Warning: Some required packages may be missing"
    echo "Please install them with: pip install torch torchvision pillow flask numpy"
    echo "And: pip install git+https://github.com/openai/CLIP.git"
    echo ""
    read -p "Do you want to install missing packages now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install torch torchvision pillow flask numpy
        pip install git+https://github.com/openai/CLIP.git
    fi
fi

echo ""
echo "Starting Flask application..."
echo "Server will be available at: http://localhost:5002"
echo "Press Ctrl+C to stop the server"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Run the application
python app.py


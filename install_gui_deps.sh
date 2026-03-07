#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting GUI dependencies installation..."

# Check if ai_env exists before trying to activate
if [ ! -d "ai_env" ]; then
    echo "Error: ai_env directory not found! Please run your main setup script first."
    exit 1
fi

# Activate the virtual environment
echo "Activating ai_env..."
source ai_env/bin/activate

# Upgrade pip just to be safe
echo "Upgrading pip..."
pip install --upgrade pip

# Install the requirements
echo "Installing libraries from requirements.txt..."
pip install -r requirements.txt

echo "Installation complete! You can now run your Streamlit apps."
echo "To run the training dashboard: streamlit run train_app.py"
echo "To run the detection dashboard: streamlit run app.py"

# Deactivate to leave the terminal in its original state
deactivate

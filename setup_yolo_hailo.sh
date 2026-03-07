#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting System Setup..."

# 1. Install System Requirements and Dependencies
echo "Installing system packages..."
sudo apt update && sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget [cite: 4]
sudo apt install htop -y [cite: 18]
sudo apt update && sudo apt install -y libbz2-dev libsqlite3-dev liblzma-dev [cite: 19]
sudo apt install linux-tools-generic hwdata -y [cite: 23]
sudo apt-get install -y cmake [cite: 133]
sudo apt-get install -y libjpeg-dev zlib1g-dev [cite: 149]

# 2. Check and Install Python 3.11.9
TARGET_PY_VERSION="3.11.9"
PYTHON_CMD="python3.11"

# Check if the python command exists and capture its version
if command -v $PYTHON_CMD &> /dev/null; then
    CURRENT_PY_VERSION=$($PYTHON_CMD -c 'import platform; print(platform.python_version())')
    if [ "$CURRENT_PY_VERSION" == "$TARGET_PY_VERSION" ]; then
        echo "✅ Python $TARGET_PY_VERSION is already installed. Skipping compilation process."
        SKIP_PYTHON=true
    else
        echo "⚠️ Found Python $CURRENT_PY_VERSION, but need $TARGET_PY_VERSION."
        SKIP_PYTHON=false
    fi
else
    echo "⚠️ $PYTHON_CMD not found."
    SKIP_PYTHON=false
fi

# Execute installation only if the check failed
if [ "$SKIP_PYTHON" = false ]; then
    echo "Downloading and installing Python 3.11.9..."
    wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz [cite: 6, 7, 8]
    tar -xf Python-3.11.9.tgz [cite: 9]
    cd Python-3.11.9 [cite: 10]
    ./configure --enable-optimizations [cite: 11]
    make -j $(nproc) [cite: 12]
    sudo make altinstall [cite: 13]
    cd ..
fi

echo "Python Version Verified:"
$PYTHON_CMD --version 

# 3. Environment Setup for YOLO Model Training
echo "Setting up ai_env for YOLO..."
python3.11 -m venv ai_env [cite: 21]
source ai_env/bin/activate [cite: 21]
pip install --upgrade pip [cite: 21]
pip install ultralytics [cite: 21]
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 [cite: 22]
deactivate
echo "ai_env setup complete."

# 4. Hailo Dataflow Compiler Environment Setup
echo "Setting up Hailo environment..."
# Make sure the user's custom hailo setup script is executable and run it
chmod +x setup_hailo_env.sh 
./setup_hailo_env.sh [cite: 131, 132]

# Assuming setup_hailo_env.sh creates 'hailo_dfc_env', activate it
source hailo_dfc_env/bin/activate [cite: 134]

# Install the Hailo Dataflow Compiler wheel (Ensure it's downloaded in the current dir)
echo "Installing Hailo Dataflow Compiler..."
pip install hailo_dataflow_compiler-*.whl [cite: 145]

# Clone and setup Hailo Model Zoo
echo "Setting up Hailo Model Zoo..."
git clone https://github.com/hailo-ai/hailo_model_zoo.git [cite: 145]
cd hailo_model_zoo [cite: 146]
pip install -e . [cite: 147]
cd ..

# Clone RasPi YOLO repository
git clone https://github.com/LukeDitria/RasPi_YOLO.git [cite: 154, 155]

deactivate
echo "Hailo compilation environment setup complete."
echo "All installations finished successfully! You are ready for compilation." [cite: 150]

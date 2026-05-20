#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting System Setup..."

# 1. Install System Requirements and Dependencies
echo "Installing system packages..."
# Consolidated all apt commands into a single, faster transaction
sudo apt update
sudo apt install -y build-essential zlib1g-dev libncurses-dev libgdbm-dev \
    libnss3-dev libssl-dev libreadline-dev libffi-dev wget htop \
    libbz2-dev libsqlite3-dev liblzma-dev linux-tools-generic hwdata \
    cmake libjpeg-dev

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
    wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
    tar -xf Python-3.11.9.tgz
    cd Python-3.11.9
    ./configure --enable-optimizations
    make -j $(nproc)
    sudo make altinstall
    cd ..
    # Clean up the large tarball to save space
    rm Python-3.11.9.tgz 
fi

echo "Python Version Verified:"
$PYTHON_CMD --version

# 3. Environment Setup for YOLO Model Training
echo "Setting up ai_env for YOLO..."
$PYTHON_CMD -m venv ai_env
source ai_env/bin/activate
pip install --upgrade pip
pip install ultralytics

# Fixed: Now fetching CUDA 12.6 to support the RTX 5070's Blackwell architecture
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

deactivate
echo "ai_env setup complete."

# 4. Hailo Dataflow Compiler Environment Setup
echo "Setting up Hailo environment (hailo_dfc_env)..."
$PYTHON_CMD -m venv hailo_dfc_env
source hailo_dfc_env/bin/activate

echo "Downloading Hailo Dataflow Compiler from GitHub Releases..."
# Added -nc (no-clobber) so it doesn't repeatedly download the wheel if you rerun the script
wget -nc https://github.com/Airium001/airium-yolo-custom-train/releases/download/v1.0.0/hailo_dataflow_compiler-5.2.0-py3-none-linux_x86_64.whl

echo "Installing Hailo Dataflow Compiler..."
# Target the specific filename instead of a wildcard to avoid accidental conflicts
pip install hailo_dataflow_compiler-5.2.0-py3-none-linux_x86_64.whl

echo "Setting up Hailo Model Zoo..."
# Prevent fatal git errors by checking if the directory already exists
if [ ! -d "hailo_model_zoo" ]; then
    git clone https://github.com/hailo-ai/hailo_model_zoo.git
fi
cd hailo_model_zoo
pip install -e .
cd ..

echo "Cloning RasPi YOLO repository..."
if [ ! -d "RasPi_YOLO" ]; then
    git clone https://github.com/LukeDitria/RasPi_YOLO.git
fi

deactivate
echo "Hailo compilation environment setup complete."
echo "All installations finished successfully! You are ready to begin bypassing your models."

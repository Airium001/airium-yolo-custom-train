#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting System Setup..."

# 1. Install System Requirements and Dependencies
echo "Installing system packages..."
sudo apt update && sudo apt install -y build-essential zlib1g-dev libncurses-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
sudo apt install htop -y
sudo apt update && sudo apt install -y libbz2-dev libsqlite3-dev liblzma-dev
sudo apt install linux-tools-generic hwdata -y
sudo apt-get install -y cmake
sudo apt-get install -y libjpeg-dev zlib1g-dev

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
fi

echo "Python Version Verified:"
$PYTHON_CMD --version

# 3. Environment Setup for YOLO Model Training
echo "Setting up ai_env for YOLO..."
python3.11 -m venv ai_env
source ai_env/bin/activate
pip install --upgrade pip
pip install ultralytics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
deactivate
echo "ai_env setup complete."

# 4. Hailo Dataflow Compiler Environment Setup
echo "Setting up Hailo environment (hailo_dfc_env)..."
python3.11 -m venv hailo_dfc_env
source hailo_dfc_env/bin/activate

# --- ADD THESE TWO LINES ---
echo "Downloading Hailo Dataflow Compiler from GitHub Releases..."
wget https://github.com/Airium001/airium-yolo-custom-train/releases/download/Hailo10H_Compiler/hailo_dataflow_compiler-5.2.0-py3-none-linux_x86_64.whl
# ---------------------------

echo "Installing Hailo Dataflow Compiler..."
pip install hailo_dataflow_compiler-*.whl

echo "Setting up Hailo Model Zoo..."
git clone https://github.com/hailo-ai/hailo_model_zoo.git
cd hailo_model_zoo
pip install -e .
cd ..

echo "Cloning RasPi YOLO repository..."
git clone https://github.com/LukeDitria/RasPi_YOLO.git

deactivate
echo "Hailo compilation environment setup complete."
echo "All installations finished successfully! You are ready for compilation."

# YOLOv8 & Hailo-10h Training and Compilation Pipeline

This repository contains a complete pipeline for training custom YOLOv8 models using a Streamlit GUI, deploying them for inference, and setting up the environment for compilation to the Hailo-10h AI accelerator.

## 🚀 Getting Started

### 1. Prerequisites
You must manually download the Hailo Dataflow Compiler wheel (`.whl` file) for Hailo-10h, X86, Linux, Python 3.11 from the Hailo Developer Zone and place it in the root directory.

### 2. System Setup
Run the main setup script to install system dependencies, Python 3.11.9 (if not present), and create the necessary virtual environments (`ai_env` and `hailo_dfc_env`):
```bash
chmod +x setup_yolo_hailo.sh
./setup_system.sh

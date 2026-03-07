#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Creating Hailo virtual environment (hailo_dfc_env)..."
python3.11 -m venv hailo_dfc_env

echo "Hailo environment created successfully."

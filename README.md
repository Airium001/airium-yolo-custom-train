# Airium YOLO Custom Train & Hailo Pipeline

This repository provides a complete, end-to-end pipeline for training custom YOLOv8 models via a web interface, testing them on various media sources, and compiling the final weights for the Hailo-10h AI accelerator (specifically for use with Raspberry Pi AI Hats).

## 🔄 The Application Process

The workflow is broken down into three main stages:

### 1. Training Phase (`train_app.py`)
The **YOLO Custom Training Dashboard** is a Streamlit GUI that simplifies the entire model training process.
* **Dataset Configuration:** Includes a built-in YAML editor to easily modify your classes and dataset paths directly from the browser.
* **Model Selection:** Choose between Nano, Small, Medium YOLOv8 base weights, or load a custom `.pt` file to resume training.
* **Live Monitoring:** As the model trains, the dashboard provides live, updating charts for `mAP50`, `Box Loss`, and `Class Loss`.
* **Export Options:** Once training finishes, you can pack the entire results folder into a ZIP file for local download, or directly authenticate and upload your `best.pt` weights to a shared Google Drive folder.

### 2. Inference & Testing Phase (`app.py`)
The **Multi-Source Detection Dashboard** allows you to test your trained models instantly.
* **Model Loading:** Enter the path to your newly trained `best.pt` file to load it into memory.
* **Flexible Inputs:** Test the model's accuracy by uploading static **Images**, running it over recorded **Videos**, or connecting to your **Live Camera** using WebRTC.
* **Adjustable Confidence:** Tweak the confidence threshold via a sidebar slider in real-time to see how the model performs under different strictness levels.

### 3. Hardware Compilation Phase (`compilation/`)
Once you are satisfied with the model's performance, the pipeline prepares it for edge deployment.
* **Calibration Data:** The `hailo_calibration_data.py` script automatically processes your dataset images (resizing and cropping) to create the exact calibration set the Hailo compiler needs.
* **Conversion:** The `.pt` file is exported to `.onnx` and then compiled into a `.hef` file using the Hailo Dataflow Compiler, optimizing it to run efficiently on the Hailo-10h neural processing unit.

---

## 🚀 Getting Started

### 1. Prerequisites
You must manually download the **Hailo Dataflow Compiler wheel (`.whl` file)** for Hailo-10h, X86, Linux, Python 3.11 from the Hailo Developer Zone and place it in the root directory.

### 2. System Setup
Run the main setup script to install system dependencies, Python 3.11.9, and create the necessary virtual environments (`ai_env` and `hailo_dfc_env`):
```bash
chmod +x setup_yolo_hailo.sh
./setup_yolo_hailo.sh

3. Install GUI Dependencies
Install the required Python libraries to run the Streamlit dashboards:

```bash
chmod +x install_gui_deps.sh
./install_gui_deps.sh'''

🖥️ Running the Dashboards
To start the Training Dashboard:

'''bash
source ai_env/bin/activate
streamlit run train_app.py'''

To start the Detection Dashboard:
'''bash
source ai_env/bin/activate
streamlit run app.py'''




=========================================
Airium YOLO Custom Train & Hailo Pipeline
=========================================

This repository provides a complete, end-to-end pipeline for training custom YOLOv8 models via a web interface, testing them on various media sources, and compiling the final weights for the Hailo-10h AI accelerator (specifically for use with Raspberry Pi AI Hats).

🔄 The Application Process
--------------------------

The workflow is broken down into three main stages:

1. Training & Inference Phase (``train_app.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The **YOLO Training & Detection Dashboard** is a unified Streamlit GUI that covers the entire
model development workflow in a single app. Use the **sidebar radio button** to switch between
two modes:

**🏋️ Train Model Mode**

* **Dataset Configuration:** Browse your project directory to auto-discover ``.yaml`` files,
  or enter a path manually. A built-in YAML editor lets you add, remove, or rename classes
  and update ``nc`` without leaving the browser.
* **Model Selection:** Choose between ``yolov8n.pt`` (Nano), ``yolov8s.pt`` (Small),
  ``yolov8m.pt`` (Medium), or provide a custom ``.pt`` path to resume training from
  existing weights.
* **Training Parameters:** Configure epochs, image size (128–1280 px), batch size, device
  (``cpu`` or GPU ``0``), optimizer (SGD / Adam / AdamW / auto), and learning rate directly
  from the sidebar.
* **Live Monitoring:** Real-time ``mAP50``, ``Box Loss``, and ``Class Loss`` charts update
  after every epoch alongside a progress bar and epoch counter.
* **Results Viewer:** After training, browse confusion matrices, F1/Precision/Recall
  confidence curves, label distribution plots, and validation batch previews — all inside
  the dashboard.
* **Export Options:** Pack the entire results folder into a ``.zip`` for local download, or
  authenticate via OAuth and upload ``best.pt`` directly to a shared Google Drive folder.
* **GPU Cleanup:** The training model is automatically unloaded from GPU memory after
  completion. A manual **Free GPU Memory** button in the sidebar is also available between
  sessions.

**🎯 Run Detection Mode**

* **Model Loading:** Enter the path to any ``.pt`` weights file and click **Load Model**, or
  load the freshly trained model directly from the training result with one click.
* **Image Detection:** Upload a ``JPG`` or ``PNG`` and run single-shot inference with
  annotated bounding box output.
* **Video Detection:** Upload an ``MP4``, ``AVI``, or ``MOV`` file and process every frame
  with live bounding box rendering.
* **Live Camera:** Stream from your browser camera via WebRTC with per-frame YOLO inference.
  A frame-skip slider (1–10) prevents stream freezing on CPU. Resolution is capped at
  640×480 and Google STUN servers are pre-configured for NAT/firewall compatibility.
* **Adjustable Confidence:** A sidebar slider controls the confidence threshold in real-time
  across all input sources.
* **GPU Cleanup:** A **Free GPU & Unload Model** button in the sidebar releases memory when
  switching between models.

2. Hardware Compilation Phase (``hailo_compile_app.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The **Hailo10h Compilation Dashboard** is a dedicated Streamlit GUI that converts your
trained ``.pt`` model into a ``.hef`` file ready to deploy on Raspberry Pi with the
Hailo-10h AI Hat. All paths default to the directory where ``hailo_compile_app.py``
is placed. The pipeline runs four sequential steps:

* **Step 1 — Export .pt to .onnx:** Runs the ``yolo`` CLI binary inside ``ai_env`` to
  export your trained model to ONNX format. Configure image size (default 512 px) and
  ONNX opset (default 11). ``CUDA_VISIBLE_DEVICES=-1`` is set automatically to avoid
  GPU conflicts. The ``.onnx`` file is saved to the ``hailo_output/`` folder.

* **Step 2 — Generate Calibration Data:** Runs ``hailo_calibration_data.py`` from the
  ``RasPi_YOLO/`` directory to resize and crop your training images into the calibration
  set the Hailo compiler requires. Default is 256 images at 640×640 px. A warning is
  shown if fewer than 200 images are provided, as this may cause quantization failures.

* **Step 3 — Parse Model:** Runs ``hailomz parse`` inside ``hailo_dfc_env`` to convert
  the ``.onnx`` into a ``.har`` intermediate file. End-node names are pre-filled with
  standard YOLOv8n defaults and can be edited for other architectures.

* **Step 4 — Compile to .hef:** Runs ``hailomz compile`` inside ``hailo_dfc_env``.
  Accepts either the ``.onnx`` directly or the pre-parsed ``.har`` from Step 3 for
  faster re-compilation. A live calibration image counter warns if the calib folder
  has too few images before compilation starts. A performance profile selector is
  provided — use ``fastest_single_control_flow`` if compilation fails with quantization
  errors on CPU. Compilation typically takes 8–10 minutes. A download button appears
  automatically when the ``.hef`` is ready.

🚀 Getting Started
------------------

1. Prerequisites
~~~~~~~~~~~~~~~~
The setup script ``setup_yolo_hailo.sh`` will automatically download the **Hailo 
Dataflow Compiler wheel (.whl file)** for Hailo-10h, X86, Linux, Python 3.11 from 
the Hailo Developer Zone and place it in the root directory.

Installation
~~~~~~~~~~~~
Clone the repository first:

.. code-block:: bash

    git clone https://github.com/Airium001/airium-yolo-custom-train.git
    cd airium-yolo-custom-train

2. System Setup
~~~~~~~~~~~~~~~
Run the main setup script to install system dependencies, Python 3.11.9, and create the necessary virtual environments (``ai_env`` and ``hailo_dfc_env``):

.. code-block:: bash

    chmod +x setup_yolo_hailo.sh
    ./setup_yolo_hailo.sh

3. Install GUI Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Install the required Python libraries to run the Streamlit dashboards:

.. code-block:: bash

    chmod +x install_gui_deps.sh
    ./install_gui_deps.sh


🖥️ Running the Dashboards
-------------------------

**To start the Training & Testing Dashboard:**

.. code-block:: bash

    source ai_env/bin/activate
    streamlit run train_app.py

4. Hailo Compilation App
~~~~~~~~~~~~~~~~~~~~~~~~
If your ``.pt`` model is ready and you want to deploy it to your Raspberry Pi with the Hailo-10h AI Hat, compile it with the following command:

.. code-block:: bash
    
    source ai_env/bin/activate
    streamlit run hailo_compile_app.py

Acknowledgements
----------------
The calibration data script used in the Hailo compilation pipeline is based on the
work of `Luke Ditria <https://github.com/LukeDitria>`_:

* **RasPi_YOLO** — https://github.com/LukeDitria/RasPi_YOLO

  A collection of tools for running YOLO models on Raspberry Pi with the Hailo AI Hat,
  including the ``hailo_calibration_data.py`` script used in Step 2 of the compilation
  pipeline.


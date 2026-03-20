import streamlit as st
import cv2
import pandas as pd
import tempfile
import os
import glob
import numpy as np
import logging
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer
import av

# Cloud download imports
import gdown
import zipfile
import requests
from roboflow import Roboflow

# Suppress harmless WebRTC/aioice ICE cleanup errors that fire on disconnect
logging.getLogger("aioice.stun").setLevel(logging.CRITICAL)
logging.getLogger("aioice.ice").setLevel(logging.CRITICAL)
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import yaml
import shutil

import gc
import torch

SCOPES = ['https://www.googleapis.com/auth/drive.file']

# ==========================================
# Page Config & Custom CSS
# ==========================================
st.set_page_config(page_title="YOLO All-in-One Dashboard", page_icon="🎯", layout="wide")

# Hide Streamlit default menus and footers, and adjust top padding
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
# Helper Functions
# ==========================================

def gpu_cleanup(label: str = ""):
    """Delete loaded model, collect garbage, and empty the CUDA cache."""
    if st.session_state.get("model") is not None:
        del st.session_state.model
        st.session_state.model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if label:
        st.toast(f"🧹 {label} — GPU/CPU memory freed.", icon="🚀")

def find_yaml_files(base_dir):
    if not os.path.exists(base_dir):
        return []
    return glob.glob(os.path.join(base_dir, "*.yaml"))

def upload_single_file_to_gdrive(file_path, new_filename, drive_folder_id, client_secret_path):
    try:
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                st.info("⚠️ Action Required: Check your terminal/browser to log in to Google!")
                flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
                creds = flow.run_local_server(port=8080)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        service = build('drive', 'v3', credentials=creds)
        file_metadata = {'name': new_filename, 'parents': [drive_folder_id]}
        media = MediaFileUpload(file_path, mimetype='application/octet-stream', resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        st.toast(f"Model successfully uploaded to Google Drive! (ID: {file.get('id')})", icon="✅")
    except Exception as e:
        st.error(f"Google Drive Upload Failed: {e}")

def create_full_zip(run_dir):
    temp_zip_base = os.path.join(tempfile.gettempdir(), "yolo_full_export")
    zip_path = shutil.make_archive(temp_zip_base, 'zip', run_dir)
    return zip_path

def web_file_browser(button_label, initial_path=".", extension="", key_prefix="fb"):
    """Creates a dropdown file navigator inside the Streamlit UI."""
    if f"{key_prefix}_path" not in st.session_state:
        st.session_state[f"{key_prefix}_path"] = os.path.abspath(initial_path)
        
    current_path = st.session_state[f"{key_prefix}_path"]
    
    with st.popover(button_label, use_container_width=True):
        st.caption(f"📂 `{current_path}`")
        
        # Up Directory Button
        if st.button("⬆️ Up One Level", key=f"{key_prefix}_up", use_container_width=True):
            st.session_state[f"{key_prefix}_path"] = os.path.dirname(current_path)
            st.rerun()
            
        try:
            items = os.listdir(current_path)
            folders = [f for f in items if os.path.isdir(os.path.join(current_path, f))]
            files = [f for f in items if os.path.isfile(os.path.join(current_path, f))]
            
            if extension:
                files = [f for f in files if f.endswith(extension)]
                
            folders.sort()
            files.sort()
            
            # Folder Navigation
            selected_folder = st.selectbox("📁 Open Folder:", ["(Stay here)"] + folders, key=f"{key_prefix}_folder")
            if selected_folder != "(Stay here)":
                st.session_state[f"{key_prefix}_path"] = os.path.join(current_path, selected_folder)
                st.rerun()
                
            # File Selection
            selected_file = st.selectbox(f"📄 Select {extension} File:", ["(None)"] + files, key=f"{key_prefix}_file")
            
            if selected_file != "(None)":
                return os.path.join(current_path, selected_file)
                
        except Exception as e:
            st.error(f"Cannot access path: {e}")
            
    return None

def extract_dataset_zip(zip_path, extract_to_folder):
    """Extracts a ZIP and searches for the dataset.yaml inside."""
    with st.spinner(f"Extracting to {extract_to_folder}..."):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_folder)
        
    yaml_files = glob.glob(os.path.join(extract_to_folder, "**", "*.yaml"), recursive=True)
    for y in yaml_files:
        if "dataset.yaml" in y.lower() or "data.yaml" in y.lower():
            return y
    return yaml_files[0] if yaml_files else None

def download_gdrive_folder(url, dest_folder="datasets/downloaded_gdrive"):
    """Downloads a folder or zip from Google Drive using gdown."""
    os.makedirs(dest_folder, exist_ok=True)
    try:
        output_path = gdown.download(url, quiet=False, fuzzy=True, output=os.path.join(dest_folder, "gdrive_dataset.zip"))
        if output_path and output_path.endswith('.zip'):
            return extract_dataset_zip(output_path, dest_folder)
        return None
    except Exception as e:
        st.error(f"Google Drive Download Failed: {e}")
        return None

# ==========================================
# Header & GPU Status
# ==========================================
st.title("🤖 YOLO All-in-One Dashboard")
st.caption("Train custom models and run real-time inference seamlessly.")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved  = torch.cuda.memory_reserved()  / 1024**2
    total     = torch.cuda.get_device_properties(0).total_memory / 1024**2
    st.info(f"🖥️ GPU: **{torch.cuda.get_device_name(0)}** | "
            f"Allocated: **{allocated:.0f} MB** | "
            f"Reserved: **{reserved:.0f} MB** | "
            f"Total: **{total:.0f} MB**")
else:
    st.info("🖥️ Running on **CPU** — no CUDA GPU detected.")

# ==========================================
# Session State Init
# ==========================================
if "last_save_dir" not in st.session_state:
    st.session_state.last_save_dir = "runs/detect/train"
if "prepared_zip_path" not in st.session_state:
    st.session_state.prepared_zip_path = None
if "model" not in st.session_state:
    st.session_state.model = None

# ==========================================
# Top-level Mode Toggle
# ==========================================
mode = st.sidebar.radio("🔀 Select Mode", ["🎯 Run Detection", "🏋️ Train Model"], index=0)
st.sidebar.divider()

# ==========================================
# ===== MODE A: DETECTION =====
# ==========================================
if mode == "🎯 Run Detection":

    with st.sidebar.container(border=True):
        st.subheader("⚙️ Model Configuration")
        
        det_model_tabs = st.tabs(["📁 Local", "☁️ GDrive / Link"])

        if "det_model_path" not in st.session_state:
            st.session_state.det_model_path = "best.pt"

        # --- TAB 1: LOCAL MODEL BROWSER ---
        with det_model_tabs[0]:
            browsed_model = web_file_browser("🔍 Browse Local .pt", extension=".pt", key_prefix="det_browser")
            if browsed_model:
                st.session_state.det_model_path = browsed_model

        # --- TAB 2: CLOUD MODEL DOWNLOAD ---
        with det_model_tabs[1]:
            st.caption("Download a .pt model from GDrive or a direct link.")
            cloud_model_url = st.text_input("Model URL (.pt file):")
            is_model_gdrive = "drive.google.com" in cloud_model_url

            if st.button("Download Model", use_container_width=True):
                if not cloud_model_url:
                    st.warning("Please enter a valid URL.")
                else:
                    with st.spinner("Downloading model..."):
                        dest_folder = os.path.join("models", "cloud_download")
                        os.makedirs(dest_folder, exist_ok=True)
                        dest_file = os.path.join(dest_folder, "downloaded_model.pt")

                        try:
                            if is_model_gdrive:
                                gdown.download(cloud_model_url, dest_file, quiet=False, fuzzy=True)
                            else:
                                r = requests.get(cloud_model_url, stream=True)
                                with open(dest_file, 'wb') as f:
                                    for chunk in r.iter_content(chunk_size=8192):
                                        f.write(chunk)

                            if os.path.exists(dest_file):
                                st.session_state.det_model_path = dest_file
                                st.toast("Cloud model downloaded!", icon="✅")
                            else:
                                st.error("Download failed.")
                        except Exception as e:
                            st.error(f"Error downloading model: {e}")

        st.divider()
        model_path_input = st.text_input("Active Model Path:", value=st.session_state.det_model_path)
        st.session_state.det_model_path = model_path_input

        if st.button("Load Model", type="primary", use_container_width=True):
            if os.path.exists(model_path_input):
                st.session_state.model = YOLO(model_path_input)
                st.toast(f"Model loaded: {model_path_input}", icon="✅")
            else:
                st.error("Model file not found!")

    with st.sidebar.container(border=True):
        st.subheader("🎛️ Detection Settings")
        confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
        source_type = st.selectbox("Input Source", ["Image", "Video", "Live Camera"])

    st.sidebar.divider()
    if st.sidebar.button("🧹 Free GPU & Unload Model", type="secondary", use_container_width=True):
        gpu_cleanup("Detection session cleared")

    st.write("### Detection Viewer")

    if st.session_state.model is None:
        st.info(
            "#### 👈 Welcome to the Detection Viewer\n"
            "To get started, enter your YOLO `.pt` model path in the sidebar and click **Load Model**."
        )
    else:
        # --- IMAGE ---
        if source_type == "Image":
            uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None and st.button("Run Detection", type="primary"):
                with st.spinner("Processing image..."):
                    image = Image.open(uploaded_image)
                    img_array = np.array(image)
                    results = st.session_state.model.predict(source=img_array, conf=confidence, device='cpu')
                    annotated_img = results[0].plot()
                    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    st.image(annotated_img, caption="Detection Results", width='stretch')

        # --- VIDEO ---
        elif source_type == "Video":
            uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
            if uploaded_video is not None and st.button("Run Detection", type="primary"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                frame_placeholder = st.empty()
                results = st.session_state.model.predict(source=tfile.name, conf=confidence, device='cpu', stream=True)
                for r in results:
                    annotated_frame = r.plot()
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(annotated_frame, channels="RGB", width='stretch')
                st.toast("Video processing complete!", icon="✅")

        # --- LIVE CAMERA ---
        elif source_type == "Live Camera":
            st.info("Click 'START' below to grant browser camera access.")
            process_every_n = st.slider("Run detection every N frames (higher = smoother, less frequent)", 1, 10, 3)

            current_model = st.session_state.model
            current_conf = confidence
            current_n = process_every_n

            frame_state = {"count": 0, "last_annotated": None}

            def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                frame_state["count"] += 1

                if frame_state["count"] % current_n == 0:
                    results = current_model.predict(source=img, conf=current_conf, device='cpu', verbose=False)
                    frame_state["last_annotated"] = results[0].plot()

                out = frame_state["last_annotated"] if frame_state["last_annotated"] is not None else img
                return av.VideoFrame.from_ndarray(out, format="bgr24")

            webrtc_streamer(
                key="yolo_detection",
                video_frame_callback=video_frame_callback,
                rtc_configuration={
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                    ]
                },
                media_stream_constraints={
                    "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                    "audio": False
                },
            )

# ==========================================
# ===== MODE B: TRAINING =====
# ==========================================
elif mode == "🏋️ Train Model":

    with st.sidebar.container(border=True):
        st.subheader("1. Dataset Configuration")
        
        data_tabs = st.tabs(["📁 Local", "🌌 Roboflow", "☁️ GDrive / Link"])
        
        if "train_yaml_path" not in st.session_state:
            st.session_state.train_yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_data", "dataset.yaml")

        with data_tabs[0]:
            browsed_yaml = web_file_browser("🔍 Browse Local Files", extension=".yaml", key_prefix="yaml_browser")
            if browsed_yaml:
                st.session_state.train_yaml_path = browsed_yaml

        with data_tabs[1]:
            st.caption("Download directly via Roboflow API")
            rf_api_key = st.text_input("API Key:", type="password")
            rf_workspace = st.text_input("Workspace Name:")
            rf_project = st.text_input("Project Name:")
            rf_version = st.number_input("Version Number:", min_value=1, step=1)
            
            if st.button("Download from Roboflow", use_container_width=True):
                if not rf_api_key or not rf_workspace or not rf_project:
                    st.warning("Please fill in all Roboflow fields.")
                else:
                    with st.spinner("Connecting to Roboflow..."):
                        try:
                            rf = Roboflow(api_key=rf_api_key)
                            project = rf.workspace(rf_workspace).project(rf_project)
                            dataset = project.version(rf_version).download("yolov8", location=f"datasets/{rf_project}_v{rf_version}")
                            
                            downloaded_yaml = os.path.join(dataset.location, "data.yaml")
                            if os.path.exists(downloaded_yaml):
                                st.session_state.train_yaml_path = downloaded_yaml
                                st.toast("Roboflow dataset ready!", icon="✅")
                            else:
                                st.error("Downloaded successfully, but couldn't locate data.yaml.")
                        except Exception as e:
                            st.error(f"Roboflow Error: {e}")

        with data_tabs[2]:
            st.caption("Paste a shared GDrive link or direct ZIP URL")
            cloud_url = st.text_input("Dataset URL:")
            is_gdrive = "drive.google.com" in cloud_url
            
            if st.button("Download & Extract", use_container_width=True):
                if not cloud_url:
                    st.warning("Please enter a valid URL.")
                else:
                    with st.spinner("Downloading dataset... this might take a minute."):
                        dest_folder = os.path.join("datasets", "cloud_download")
                        os.makedirs(dest_folder, exist_ok=True)
                        
                        found_yaml = None
                        if is_gdrive:
                            found_yaml = download_gdrive_folder(cloud_url, dest_folder)
                        else:
                            try:
                                local_zip = os.path.join(dest_folder, "download.zip")
                                r = requests.get(cloud_url, stream=True)
                                with open(local_zip, 'wb') as f:
                                    for chunk in r.iter_content(chunk_size=8192): 
                                        f.write(chunk)
                                found_yaml = extract_dataset_zip(local_zip, dest_folder)
                            except Exception as e:
                                st.error(f"Direct download failed: {e}")
                                
                        if found_yaml:
                            st.session_state.train_yaml_path = found_yaml
                            st.toast("Cloud dataset downloaded and extracted!", icon="✅")
                        else:
                            st.error("Download finished, but no .yaml file was found in the extracted folder.")

        st.divider()
        yaml_path_input = st.text_input("Active Target YAML Path:", value=st.session_state.train_yaml_path)
        st.session_state.train_yaml_path = yaml_path_input


    with st.sidebar.container(border=True):
        st.subheader("2. Base Model Selection")
        model_type = st.selectbox("Select Model Weights", [
            "yolov8n.pt (Nano - Fastest)",
            "yolov8s.pt (Small - Better Accuracy)",
            "yolov8m.pt (Medium - Balanced)",
            "Custom Path..."
        ])
        
        if model_type == "Custom Path...":
            if "custom_base_pt" not in st.session_state:
                st.session_state.custom_base_pt = "yolov8n.pt"
                
            browsed_base_pt = web_file_browser("🔍 Browse for Base Model", extension=".pt", key_prefix="base_pt_browser")
            if browsed_base_pt:
                st.session_state.custom_base_pt = browsed_base_pt
                
            model_path = st.text_input("Custom .pt path:", value=st.session_state.custom_base_pt)
        else:
            model_path = model_type.split(" ")[0]

    with st.sidebar.container(border=True):
        st.subheader("3. Training Parameters")
        epochs = st.number_input("Epochs", min_value=1, value=10)
        imgsz = st.slider("Image Size", min_value=128, max_value=1280, step=32, value=224)
        batch_size = st.selectbox("Batch Size", [1, 2, 4, 6, 8, 16, 32, 64, -1], index=1)
        device_opt = st.selectbox("Device", ["cpu", "0"], index=0)
        col_opt, col_lr = st.columns(2)
        with col_opt:
            optimizer = st.selectbox("Optimizer", ["auto", "SGD", "Adam", "AdamW"], index=0)
        with col_lr:
            learning_rate = st.number_input("Learning Rate", value=0.01, format="%.4f")

    st.sidebar.divider()
    if st.sidebar.button("🧹 Free GPU Memory", type="secondary", use_container_width=True, help="Run this between training sessions."):
        gpu_cleanup("Training session cleared")

    # --- Class Editor ---
    with st.expander("📝 Dataset Class Editor (dataset.yaml)", expanded=False):
        if os.path.exists(yaml_path_input):
            try:
                with open(yaml_path_input, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                current_classes = yaml_data.get('names', [])
                if isinstance(current_classes, dict):
                    current_classes = list(current_classes.values())
                st.write(f"Current `nc`: **{yaml_data.get('nc', len(current_classes))}**")
                new_classes_str = st.text_area("Edit Classes (Comma-Separated):", value=", ".join(current_classes))
                if st.button("Save Changes to YAML"):
                    new_list = [c.strip() for c in new_classes_str.split(",") if c.strip()]
                    yaml_data['names'], yaml_data['nc'] = new_list, len(new_list)
                    with open(yaml_path_input, 'w') as f:
                        yaml.dump(yaml_data, f, sort_keys=False)
                    st.toast("YAML Updated Successfully!", icon="✅")
                    st.rerun()
            except Exception as e:
                st.error(f"Error reading YAML: {e}")

    st.divider()

    # --- Training Execution ---
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        if not os.path.exists(yaml_path_input) or (model_type == "Custom Path..." and not os.path.exists(model_path)):
            st.error("ERROR: Check dataset or model paths.")
        else:
            st.write("### Live Training Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**mAP50**")
                map_chart = st.empty()
            with col2:
                st.write("**Box Loss**")
                loss_chart = st.empty()
            with col3:
                st.write("**Class Loss**")
                cls_loss_chart = st.empty()

            history = []
            model = YOLO(model_path)

            def on_train_epoch_end(trainer):
                curr_ep, tot_ep = trainer.epoch + 1, trainer.epochs
                progress_bar.progress(curr_ep / tot_ep)
                status_text.text(f"Processing Epoch {curr_ep} of {tot_ep}...")
                if trainer.metrics:
                    history.append({
                        "Epoch": curr_ep,
                        "mAP50": trainer.metrics.get("metrics/mAP50(B)", 0),
                        "Box Loss": trainer.metrics.get("val/box_loss", 0),
                        "Class Loss": trainer.metrics.get("val/cls_loss", 0)
                    })
                    df = pd.DataFrame(history).set_index("Epoch")
                    map_chart.line_chart(df[["mAP50"]])
                    loss_chart.line_chart(df[["Box Loss"]])
                    cls_loss_chart.line_chart(df[["Class Loss"]])

            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            status_text.text("Initializing training... please wait.")
            results = model.train(
                data=yaml_path_input, epochs=epochs, imgsz=imgsz,
                batch=batch_size, device=device_opt, optimizer=optimizer,
                lr0=learning_rate, plots=True
            )
            st.session_state.last_save_dir = str(results.save_dir)
            st.session_state.prepared_zip_path = None
            st.success(f"Training Complete! Saved to: {results.save_dir}")

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            st.toast("🧹 Training model unloaded from GPU memory.")

            trained_weights = os.path.join(str(results.save_dir), "weights", "best.pt")
            if os.path.exists(trained_weights):
                if st.button("🎯 Load this model into Detector", use_container_width=True):
                    st.session_state.model = YOLO(trained_weights)
                    st.toast("Model loaded! Switch to 'Run Detection' mode in the sidebar.", icon="✅")

    st.divider()

    # --- Results & Export ---
    st.write("### 📊 View Results & Export")
    target_dir = st.text_input("Target Run Directory:", value=st.session_state.last_save_dir)

    if os.path.exists(target_dir):
        tab_metrics, tab_val, tab_local, tab_drive = st.tabs([
            "📊 Metrics & Plots",
            "🖼️ Validation Previews",
            "💻 Local ZIP",
            "☁️ GDrive Upload"
        ])

        with tab_metrics:
            st.markdown("#### 📈 Training Overview")
            res_img = os.path.join(target_dir, "results.png")
            if os.path.exists(res_img):
                st.image(res_img, caption="Losses and Metrics Timeline", width='stretch')

            st.markdown("#### 🎯 Confusion Matrices")
            cm_col1, cm_col2 = st.columns(2)
            with cm_col1:
                mat_img = os.path.join(target_dir, "confusion_matrix.png")
                if os.path.exists(mat_img):
                    st.image(mat_img, caption="Absolute Confusion Matrix", width='stretch')
            with cm_col2:
                mat_norm_img = os.path.join(target_dir, "confusion_matrix_normalized.png")
                if os.path.exists(mat_norm_img):
                    st.image(mat_norm_img, caption="Normalized Confusion Matrix", width='stretch')

            st.markdown("#### 📊 Confidence Curves")
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                f1_img = os.path.join(target_dir, "BoxF1_curve.png")
                if os.path.exists(f1_img):
                    st.image(f1_img, caption="F1-Confidence Curve", width='stretch')
                p_img = os.path.join(target_dir, "BoxP_curve.png")
                if os.path.exists(p_img):
                    st.image(p_img, caption="Precision-Confidence Curve", width='stretch')
            with c_col2:
                pr_img = os.path.join(target_dir, "BoxPR_curve.png")
                if os.path.exists(pr_img):
                    st.image(pr_img, caption="Precision-Recall Curve", width='stretch')
                r_img = os.path.join(target_dir, "BoxR_curve.png")
                if os.path.exists(r_img):
                    st.image(r_img, caption="Recall-Confidence Curve", width='stretch')

            st.markdown("#### 🏷️ Dataset Labels Analysis")
            l_col1, l_col2 = st.columns(2)
            with l_col1:
                labels_img = os.path.join(target_dir, "labels.jpg")
                if os.path.exists(labels_img):
                    st.image(labels_img, caption="Dataset Labels Overview", width='stretch')
            with l_col2:
                labels_cor_img = os.path.join(target_dir, "labels_correlogram.jpg")
                if os.path.exists(labels_cor_img):
                    st.image(labels_cor_img, caption="Labels Correlogram", width='stretch')

        with tab_val:
            st.caption("Batch prediction images generated during the validation phase.")
            val_images = [f for f in os.listdir(target_dir) if f.startswith('val_batch') and f.endswith('.jpg') and 'pred' in f]
            if val_images:
                val_images.sort()
                cols = st.columns(2)
                for i, img_name in enumerate(val_images):
                    img_path = os.path.join(target_dir, img_name)
                    cols[i % 2].image(img_path, caption=img_name, width='stretch')
            else:
                st.info("No validation predictions found in this directory.")

        with tab_local:
            st.caption("Packs the entire results folder into a ZIP for local download.")
            zip_rename = st.text_input("Rename ZIP file to:", value="full_training_results.zip")
            if not zip_rename.endswith(".zip"):
                zip_rename += ".zip"
            col_pack, col_down = st.columns(2)
            with col_pack:
                if st.button("1. Pack Directory", use_container_width=True):
                    with st.spinner("Zipping files..."):
                        st.session_state.prepared_zip_path = create_full_zip(target_dir)
                    st.success("Packed! Ready for download.")
            with col_down:
                if st.session_state.prepared_zip_path and os.path.exists(st.session_state.prepared_zip_path):
                    with open(st.session_state.prepared_zip_path, "rb") as f:
                        st.download_button(
                            label="2. ⬇️ Download ZIP",
                            data=f,
                            file_name=zip_rename,
                            mime="application/zip",
                            type="primary",
                            use_container_width=True
                        )

        with tab_drive:
            st.caption("Uploads ONLY the weights (best.pt) to Google Drive.")
            weights_file = os.path.join(target_dir, "weights", "best.pt")
            if not os.path.exists(weights_file):
                st.error(f"Could not find best.pt in {weights_file}")
            else:
                pt_rename = st.text_input("Rename model file to:", value="my_best_model.pt")
                if not pt_rename.endswith(".pt"):
                    pt_rename += ".pt"
                gdrive_link = st.text_input("Paste Shared Google Drive Folder Link:")
                json_key_path = st.text_input("Local path to JSON Key file:", value="credentials.json")
                if st.button("Upload Model", type="primary", use_container_width=True):
                    if not gdrive_link:
                        st.warning("Please provide the Google Drive folder link.")
                    elif not os.path.exists(json_key_path):
                        st.error(f"Could not find the JSON key at: {json_key_path}.")
                    else:
                        with st.spinner("Authenticating and uploading to Google Drive..."):
                            folder_id = gdrive_link.split("/")[-1].split("?")[0]
                            upload_single_file_to_gdrive(weights_file, pt_rename, folder_id, json_key_path)
    else:
        st.info("Enter a valid training directory path to view past results and export them.")

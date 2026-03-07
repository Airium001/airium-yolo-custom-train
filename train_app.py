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
        st.success(f"🧹 {label} — GPU/CPU memory freed. Ready for a fresh run.")


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
        st.info("☁️ Authenticated successfully! Connecting to Drive...")
        service = build('drive', 'v3', credentials=creds)
        st.info(f"🚀 Uploading {new_filename}...")
        file_metadata = {'name': new_filename, 'parents': [drive_folder_id]}
        media = MediaFileUpload(file_path, mimetype='application/octet-stream', resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        st.success(f"✅ Model successfully uploaded to Google Drive! (File ID: {file.get('id')})")
    except Exception as e:
        st.error(f"Google Drive Upload Failed: {e}")

def create_full_zip(run_dir):
    temp_zip_base = os.path.join(tempfile.gettempdir(), "yolo_full_export")
    zip_path = shutil.make_archive(temp_zip_base, 'zip', run_dir)
    return zip_path


# ==========================================
# Page Config
# ==========================================
st.set_page_config(page_title="YOLO All-in-One Dashboard", layout="wide")
st.title("🤖 YOLO All-in-One Dashboard")
st.caption("Train custom models, then run detection — all in one place.")

# GPU status badge
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

    st.sidebar.header("Model Configuration")
    model_path_input = st.sidebar.text_input("Enter Model Path:", "best.pt")

    if st.sidebar.button("Load Model"):
        if os.path.exists(model_path_input):
            st.session_state.model = YOLO(model_path_input)
            st.sidebar.success(f"Loaded: {model_path_input}")
        else:
            st.sidebar.error("Model file not found!")

    st.sidebar.header("Detection Settings")
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
    source_type = st.sidebar.selectbox("Input Source", ["Image", "Video", "Live Camera"])

    st.sidebar.divider()
    st.sidebar.header("🧹 Cleanup")
    if st.sidebar.button("Free GPU & Unload Model", type="secondary"):
        gpu_cleanup("Detection session cleared")

    st.write("### Detection Viewer")

    if st.session_state.model is None:
        st.warning("👈 Please enter your model path and click 'Load Model' in the sidebar to begin.")
    else:
        # --- IMAGE ---
        if source_type == "Image":
            uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None and st.button("Run Detection"):
                image = Image.open(uploaded_image)
                img_array = np.array(image)
                results = st.session_state.model.predict(source=img_array, conf=confidence, device='cpu')
                annotated_img = results[0].plot()
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_img, caption="Detection Results", width='stretch')

        # --- VIDEO ---
        elif source_type == "Video":
            uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
            if uploaded_video is not None and st.button("Run Detection"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                frame_placeholder = st.empty()
                results = st.session_state.model.predict(source=tfile.name, conf=confidence, device='cpu', stream=True)
                for r in results:
                    annotated_frame = r.plot()
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(annotated_frame, channels="RGB", width='stretch')
                st.success("Video processing complete!")

        # --- LIVE CAMERA ---
        elif source_type == "Live Camera":
            st.info("Click 'START' below to grant browser camera access.")

            # Frame-skip slider: only run YOLO every N frames to prevent freeze on CPU
            process_every_n = st.slider("Run detection every N frames (higher = smoother, less frequent)", 1, 10, 3)

            current_model = st.session_state.model
            current_conf = confidence
            current_n = process_every_n

            # Counter lives outside the callback via a mutable container
            frame_state = {"count": 0, "last_annotated": None}

            def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                frame_state["count"] += 1

                if frame_state["count"] % current_n == 0:
                    results = current_model.predict(source=img, conf=current_conf, device='cpu', verbose=False)
                    frame_state["last_annotated"] = results[0].plot()

                # Return last annotated frame, or raw frame if detection hasn't run yet
                out = frame_state["last_annotated"] if frame_state["last_annotated"] is not None else img
                return av.VideoFrame.from_ndarray(out, format="bgr24")

            webrtc_streamer(
                key="yolo_detection",
                video_frame_callback=video_frame_callback,
                # Public STUN servers so connection works behind NAT/firewalls
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

    # --- Sidebar: Dataset ---
    st.sidebar.header("1. Dataset Configuration")
    dataset_search_dir = st.sidebar.text_input("📂 Browse Directory for YAML:", value=os.path.dirname(os.path.abspath(__file__)))
    available_yamls = find_yaml_files(dataset_search_dir)
    available_yamls.insert(0, "Manual Entry...")
    selected_yaml = st.sidebar.selectbox("📄 Select Dataset YAML:", available_yamls)
    if selected_yaml == "Manual Entry...":
        default_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_data", "dataset.yaml")
        yaml_path_input = st.sidebar.text_input("Path to dataset.yaml", value=default_yaml)
    else:
        yaml_path_input = selected_yaml

    # --- Sidebar: Base Model ---
    st.sidebar.header("2. Base Model Selection")
    model_type = st.sidebar.selectbox("Select Model Weights", [
        "yolov8n.pt (Nano - Fastest)",
        "yolov8s.pt (Small - Better Accuracy)",
        "yolov8m.pt (Medium - Balanced)",
        "Custom Path..."
    ])
    model_path = st.sidebar.text_input("Enter custom .pt path:", "runs/detect/train/weights/best.pt") \
        if model_type == "Custom Path..." else model_type.split(" ")[0]

    # --- Sidebar: Training Params ---
    st.sidebar.header("3. Training Parameters")
    epochs = st.sidebar.number_input("Epochs", min_value=1, value=10)
    imgsz = st.sidebar.slider("Image Size", min_value=128, max_value=1280, step=32, value=224)
    batch_size = st.sidebar.selectbox("Batch Size", [1, 2, 4, 6, 8, 16, 32, 64, -1], index=1)
    device_opt = st.sidebar.selectbox("Device", ["cpu", "0"], index=0)
    optimizer = st.sidebar.selectbox("Optimizer", ["auto", "SGD", "Adam", "AdamW"], index=0)
    learning_rate = st.sidebar.number_input("Learning Rate (lr0)", value=0.01, format="%.4f")

    st.sidebar.divider()
    st.sidebar.header("🧹 Cleanup")
    st.sidebar.caption("Run this between training sessions to fully free GPU memory.")
    if st.sidebar.button("Free GPU Memory", type="secondary"):
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
                    st.success("Updated!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error reading YAML: {e}")

    st.divider()

    # --- Training Execution ---
    if st.button("🚀 Start Training", type="primary"):
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

            # Free the training model from GPU immediately after training
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            st.info("🧹 Training model unloaded from GPU memory.")

            # Auto-offer: load the freshly trained model into the detector
            trained_weights = os.path.join(str(results.save_dir), "weights", "best.pt")
            if os.path.exists(trained_weights):
                if st.button("🎯 Load this model into Detector"):
                    st.session_state.model = YOLO(trained_weights)
                    st.success("Model loaded! Switch to 'Run Detection' mode in the sidebar.")

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
                if st.button("1. Pack Directory"):
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
                            type="primary"
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
                if st.button("Upload Model", type="primary"):
                    if not gdrive_link:
                        st.warning("Please provide the Google Drive folder link.")
                    elif not os.path.exists(json_key_path):
                        st.error(f"Could not find the JSON key at: {json_key_path}.")
                    else:
                        folder_id = gdrive_link.split("/")[-1].split("?")[0]
                        upload_single_file_to_gdrive(weights_file, pt_rename, folder_id, json_key_path)
    else:
        st.info("Enter a valid training directory path to view past results and export them.")

import streamlit as st
import pandas as pd
import os
import glob
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
import yaml
from ultralytics import YOLO
import shutil
import tempfile

# This tells Google we only want permission to upload files, not delete them
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# --- Helper: Find YAML files ---
def find_yaml_files(base_dir):
    """Searches a directory for .yaml files."""
    if not os.path.exists(base_dir):
        return []
    search_pattern = os.path.join(base_dir, "*.yaml")
    return glob.glob(search_pattern)

# --- Google Drive OAuth Upload Function ---
def upload_single_file_to_gdrive(file_path, new_filename, drive_folder_id, client_secret_path):
    try:
        creds = None
        # The script will save a token.json file so you only have to log in once
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            
        # If there are no valid credentials, force the user to log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                st.info("⚠️ Action Required: Check your terminal/browser to log in to Google!")
                flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
                # Opens a browser window to ask for your permission
                creds = flow.run_local_server(port=8080)
                
            # Save the credentials for the next run
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

# --- Local Full Zip Function ---
def create_full_zip(run_dir):
    # Creates a zip of the entire folder in your system's temporary directory
    temp_zip_base = os.path.join(tempfile.gettempdir(), "yolo_full_export")
    zip_path = shutil.make_archive(temp_zip_base, 'zip', run_dir)
    return zip_path


# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="YOLOv8 Training Dashboard", layout="wide")
st.title("YOLO Custom Training Dashboard")

# Initialize session state variables
if "last_save_dir" not in st.session_state:
    st.session_state.last_save_dir = "runs/detect/train"
if "prepared_zip_path" not in st.session_state:
    st.session_state.prepared_zip_path = None

# ==========================================
# 2. Sidebar Setup
# ==========================================
st.sidebar.header("1. Dataset Configuration")

# Dataset Browser
dataset_search_dir = st.sidebar.text_input("📂 Browse Directory for YAML:", value=os.path.dirname(os.path.abspath(__file__)))
available_yamls = find_yaml_files(dataset_search_dir)
available_yamls.insert(0, "Manual Entry...")

selected_yaml = st.sidebar.selectbox("📄 Select Dataset YAML:", available_yamls)

if selected_yaml == "Manual Entry...":
    default_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_data", "dataset.yaml")
    yaml_path_input = st.sidebar.text_input("Path to dataset.yaml", value=default_yaml)
else:
    yaml_path_input = selected_yaml


st.sidebar.header("2. Base Model Selection")
model_type = st.sidebar.selectbox("Select Model Weights", ["yolov8n.pt (Nano - Fastest)", "yolov8s.pt (Small - Better Accuracy)", "yolov8m.pt (Medium - Balanced)", "Custom Path..."])
model_path = st.sidebar.text_input("Enter custom .pt path:", "runs/detect/train/weights/best.pt") if model_type == "Custom Path..." else model_type.split(" ")[0]


st.sidebar.header("3. Training Parameters")
epochs = st.sidebar.number_input("Epochs", min_value=1, value=10)
imgsz = st.sidebar.slider("Image Size", min_value=128, max_value=1280, step=32, value=224)
batch_size = st.sidebar.selectbox("Batch Size", [1,2,4,6,8, 16, 32, 64, -1], index=1)
device_opt = st.sidebar.selectbox("Device", ["cpu", "0"], index=0)
optimizer = st.sidebar.selectbox("Optimizer", ["auto", "SGD", "Adam", "AdamW"], index=0)
learning_rate = st.sidebar.number_input("Learning Rate (lr0)", value=0.01, format="%.4f")


# ==========================================
# 3. Main UI: Dataset Class Editor
# ==========================================
with st.expander("📝 Dataset Class Editor (dataset.yaml)", expanded=False):
    if os.path.exists(yaml_path_input):
        try:
            with open(yaml_path_input, 'r') as f: yaml_data = yaml.safe_load(f)
            current_classes = yaml_data.get('names', [])
            if isinstance(current_classes, dict): current_classes = list(current_classes.values())
            st.write(f"Current `nc`: **{yaml_data.get('nc', len(current_classes))}**")
            new_classes_str = st.text_area("Edit Classes (Comma-Separated):", value=", ".join(current_classes))
            if st.button("Save Changes to YAML"):
                new_list = [c.strip() for c in new_classes_str.split(",") if c.strip()]
                yaml_data['names'], yaml_data['nc'] = new_list, len(new_list)
                with open(yaml_path_input, 'w') as f: yaml.dump(yaml_data, f, sort_keys=False)
                st.success("Updated!"); st.rerun() 
        except Exception as e: st.error(f"Error reading YAML: {e}")

st.divider()

# ==========================================
# 4. Main Training Execution
# ==========================================
if st.button("🚀 Start Training", type="primary"):
    if not os.path.exists(yaml_path_input) or (model_type == "Custom Path..." and not os.path.exists(model_path)):
        st.error("ERROR: Check dataset or model paths.")
    else:
        st.write("### Live Training Progress")
        progress_bar, status_text = st.progress(0), st.empty()
        col1, col2, col3 = st.columns(3)
        with col1: st.write("**mAP50**"); map_chart = st.empty()
        with col2: st.write("**Box Loss**"); loss_chart = st.empty()
        with col3: st.write("**Class Loss**"); cls_loss_chart = st.empty()
            
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
        results = model.train(data=yaml_path_input, epochs=epochs, imgsz=imgsz, batch=batch_size, device=device_opt, optimizer=optimizer, lr0=learning_rate, plots=True)
        
        st.session_state.last_save_dir = results.save_dir
        # Reset the prepared zip if a new model is trained
        st.session_state.prepared_zip_path = None 
        st.success(f"Training Complete! Saved to: {results.save_dir}")

st.divider()

# ==========================================
# 5. View Results & Export
# ==========================================
st.write("### 📊 View Results & Export")

target_dir = st.text_input("Target Run Directory:", value=st.session_state.last_save_dir)

if os.path.exists(target_dir):
    
    # Clean UI grouping using tabs
    tab_metrics, tab_val, tab_local, tab_drive = st.tabs([
        "📊 Metrics & Plots", 
        "🖼️ Validation Previews", 
        "💻 Local ZIP", 
        "☁️ GDrive Upload"
    ])
    
    # --- TAB 1: METRICS & PLOTS ---
    with tab_metrics:
        st.markdown("#### 📈 Training Overview")
        res_img = os.path.join(target_dir, "results.png")
        if os.path.exists(res_img): 
            st.image(res_img, caption="Losses and Metrics Timeline", use_container_width=True)

        st.markdown("#### 🎯 Confusion Matrices")
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            mat_img = os.path.join(target_dir, "confusion_matrix.png")
            if os.path.exists(mat_img): st.image(mat_img, caption="Absolute Confusion Matrix", use_container_width=True)
        with cm_col2:
            mat_norm_img = os.path.join(target_dir, "confusion_matrix_normalized.png")
            if os.path.exists(mat_norm_img): st.image(mat_norm_img, caption="Normalized Confusion Matrix", use_container_width=True)

        st.markdown("#### 📊 Confidence Curves")
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            f1_img = os.path.join(target_dir, "BoxF1_curve.png")
            if os.path.exists(f1_img): st.image(f1_img, caption="F1-Confidence Curve", use_container_width=True)
            p_img = os.path.join(target_dir, "BoxP_curve.png")
            if os.path.exists(p_img): st.image(p_img, caption="Precision-Confidence Curve", use_container_width=True)
        with c_col2:
            pr_img = os.path.join(target_dir, "BoxPR_curve.png")
            if os.path.exists(pr_img): st.image(pr_img, caption="Precision-Recall Curve", use_container_width=True)
            r_img = os.path.join(target_dir, "BoxR_curve.png")
            if os.path.exists(r_img): st.image(r_img, caption="Recall-Confidence Curve", use_container_width=True)

        st.markdown("#### 🏷️ Dataset Labels Analysis")
        l_col1, l_col2 = st.columns(2)
        with l_col1:
            labels_img = os.path.join(target_dir, "labels.jpg")
            if os.path.exists(labels_img): st.image(labels_img, caption="Dataset Labels Overview (Bounding Box Distributions)", use_container_width=True)
        with l_col2:
            labels_cor_img = os.path.join(target_dir, "labels_correlogram.jpg")
            if os.path.exists(labels_cor_img): st.image(labels_cor_img, caption="Labels Correlogram (X, Y, Width, Height)", use_container_width=True)

    # --- TAB 2: VALIDATION PREVIEWS ---
    with tab_val:
        st.caption("Batch prediction images generated during the validation phase.")
        # YOLO saves validation batches as 'val_batchX_pred.jpg' and 'val_batchX_labels.jpg'
        val_images = [f for f in os.listdir(target_dir) if f.startswith('val_batch') and f.endswith('.jpg') and 'pred' in f]
        
        if val_images:
            val_images.sort() # Ensure they are in order
            cols = st.columns(2)
            for i, img_name in enumerate(val_images):
                img_path = os.path.join(target_dir, img_name)
                cols[i % 2].image(img_path, caption=img_name, use_container_width=True)
        else:
            st.info("No validation predictions found in this directory. If training just finished, ensure 'plots=True' was enabled.")

    # --- TAB 3: LOCAL FULL ZIP ---
    with tab_local:
        st.caption("Packs the entire results folder (weights, images, metrics) into a ZIP for local download.")
        zip_rename = st.text_input("Rename ZIP file to:", value="full_training_results.zip")
        if not zip_rename.endswith(".zip"): zip_rename += ".zip"
        
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

    # --- TAB 4: GOOGLE DRIVE UPLOAD ---
    with tab_drive:
        st.caption("Uploads ONLY the weights (best.pt) to Google Drive.")
        weights_file = os.path.join(target_dir, "weights", "best.pt")
        
        if not os.path.exists(weights_file):
            st.error(f"Could not find best.pt in {weights_file}")
        else:
            pt_rename = st.text_input("Rename model file to:", value="my_best_model.pt")
            if not pt_rename.endswith(".pt"): pt_rename += ".pt"
            
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

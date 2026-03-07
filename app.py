import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer
import av

# 1. Page Configuration
st.set_page_config(page_title="Universal YOLO Dashboard", layout="wide")
st.title("Multi-Source Detection Dashboard")

# 2. Sidebar: Model Loading Widget
st.sidebar.header("1. Model Configuration")
model_path_input = st.sidebar.text_input("Enter Model Path:", "best.pt")

# Initialize session state to keep the model loaded across different clicks
if "model" not in st.session_state:
    st.session_state.model = None

if st.sidebar.button("Load Model"):
    if os.path.exists(model_path_input):
        st.session_state.model = YOLO(model_path_input)
        st.sidebar.success(f"Successfully loaded: {model_path_input}")
    else:
        st.sidebar.error("Model file not found!")

# 3. Sidebar: Detection Settings
st.sidebar.header("2. Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
source_type = st.sidebar.selectbox("Select Input Source:", ["Image", "Video", "Live Camera"])

# 4. Main Detection Widget
st.write("### Detection Viewer")

# Require the user to load a model first
if st.session_state.model is None:
    st.warning("👈 Please enter your model path and click 'Load Model' in the sidebar to begin.")
else:
    # --- OPTION A: IMAGE DETECTION ---
    if source_type == "Image":
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None and st.button("Run Detection"):
            # Convert uploaded file to an array YOLO can read
            image = Image.open(uploaded_image)
            img_array = np.array(image)
            
            # Run prediction (change device=0 if your GPU is fully configured)
            results = st.session_state.model.predict(source=img_array, conf=confidence, device='cpu')
            
            # Draw boxes and convert color formatting for Streamlit
            annotated_img = results[0].plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            st.image(annotated_img, caption="Detection Results", use_container_width=True)

    # --- OPTION B: VIDEO DETECTION ---
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
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                
            st.success("Video processing complete!")

    # --- OPTION C: LIVE CAMERA DETECTION ---
    # --- OPTION C: LIVE CAMERA (WEBRTC) ---
    elif source_type == "Live Camera":
        st.info("Click 'START' below to grant browser camera access.")
        
        # 1. Capture the Streamlit variables OUTSIDE the background thread
        current_model = st.session_state.model
        current_conf = confidence
        
        # 2. The callback function (runs in the background thread)
        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            
            # Use the captured 'current_model' and 'current_conf' instead of st.session_state
            results = current_model.predict(source=img, conf=current_conf, device='cpu', verbose=False)
            annotated_img = results[0].plot()
            
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

        # 3. Launch the streamer
        webrtc_streamer(
            key="yolo_detection", 
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False}
        )
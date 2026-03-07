import streamlit as st
import subprocess
import os
import sys
import glob
import threading
import queue
import shutil
import tempfile
import time

# ==========================================
# Base directory — all paths are relative to this script's location
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# Page Config
# ==========================================
st.set_page_config(
    page_title="Hailo10h Compiler",
    page_icon="⚡",
    layout="wide"
)

# ==========================================
# Custom CSS — white minimal theme
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #ffffff;
    color: #1a1a1a;
}
h1, h2, h3 {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    color: #111111;
}

.stApp { background-color: #f9f9f9; }

/* Step cards */
.step-card {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-left: 3px solid #c0c0c0;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.step-card.done   { border-left-color: #2e7d32; background: #f6fff6; }
.step-card.error  { border-left-color: #c62828; background: #fff6f6; }
.step-card.active { border-left-color: #e65100; background: #fffaf6; }

.step-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #555555;
    margin-bottom: 0.25rem;
}
.step-card.done  .step-title { color: #2e7d32; }
.step-card.error .step-title { color: #c62828; }
.step-card.active .step-title { color: #e65100; }

/* Terminal / log box */
.terminal {
    background: #f4f4f4;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 1rem 1.1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.76rem;
    color: #2a2a2a;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 400px;
    overflow-y: auto;
    line-height: 1.6;
}

/* Buttons */
.stButton > button {
    background: #ffffff;
    border: 1px solid #d0d0d0;
    color: #1a1a1a;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    border-radius: 5px;
    padding: 0.4rem 1rem;
    transition: all 0.15s ease;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}
.stButton > button:hover {
    background: #f0f0f0;
    border-color: #aaaaaa;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}

/* Sidebar */
div[data-testid="stSidebarContent"] {
    background-color: #fafafa;
    border-right: 1px solid #ebebeb;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# Session State
# ==========================================
defaults = {
    "step_status": {1: "idle", 2: "idle", 3: "idle", 4: "idle"},
    "log_1": "", "log_2": "", "log_3": "", "log_4": "",
    "onnx_path": "",
    "har_path": "",
    "hef_path": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================
# Helpers
# ==========================================

def run_cmd(cmd: list[str], env: dict = None) -> tuple[int, str]:
    """Run a subprocess, stream output, return (returncode, full_log)."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=merged_env,
    )
    lines = []
    for line in proc.stdout:
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)


def status_card(step_num: int, title: str, description: str):
    status = st.session_state.step_status[step_num]
    css_class = {"idle": "step-card", "active": "step-card active",
                 "done": "step-card done", "error": "step-card error"}[status]
    icon = {"idle": "○", "active": "◎", "done": "✓", "error": "✗"}[status]
    st.markdown(f"""
    <div class="{css_class}">
        <div class="step-title">{icon} Step {step_num} — {title}</div>
        <div style="font-size:0.83rem;color:#7a8797;">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def show_log(step_num: int):
    log = st.session_state[f"log_{step_num}"]
    if log:
        st.markdown(f'<div class="terminal">{log}</div>', unsafe_allow_html=True)


def python_in_venv(venv_path: str) -> str:
    """Return the python executable inside a venv."""
    return os.path.join(venv_path, "bin", "python")


def activated_env(venv_path: str) -> dict:
    """Return env vars that simulate 'source venv/bin/activate'."""
    bin_dir = os.path.join(venv_path, "bin")
    return {
        "VIRTUAL_ENV": venv_path,
        "PATH": bin_dir + ":" + os.environ.get("PATH", ""),
    }

# ==========================================
# Sidebar — global config
# ==========================================
st.sidebar.markdown("## ⚡ Hailo10h Compiler")
st.sidebar.markdown("---")

st.sidebar.header("Environments")
ai_env_path   = st.sidebar.text_input("ai_env path (YOLO/export)",
                                       value=os.path.join(BASE_DIR, "ai_env"))
hailo_env_path = st.sidebar.text_input("hailo_dfc_env path (compilation)",
                                        value=os.path.join(BASE_DIR, "hailo_dfc_env"))

st.sidebar.header("Model")
pt_model_path = st.sidebar.text_input("Source .pt model path", value=os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt"))

st.sidebar.header("Hardware")
hw_arch       = st.sidebar.selectbox("HW Architecture", ["hailo10h", "hailo8", "hailo8l"], index=0)
num_classes   = st.sidebar.number_input("Number of classes", min_value=1, value=4)

st.sidebar.header("Output")
output_dir    = st.sidebar.text_input("Output directory", value=os.path.join(BASE_DIR, "hailo_output"))

st.sidebar.markdown("---")
st.sidebar.caption("Pipeline runs sequentially. Complete each step before proceeding.")

# ==========================================
# Main layout
# ==========================================
st.markdown("# ⚡ Hailo10h Compilation Pipeline")
st.markdown("Convert a trained YOLOv8 `.pt` model all the way to a `.hef` file ready for Raspberry Pi + Hailo10h.")
st.markdown("---")

# Pipeline overview
col_a, col_b, col_c, col_d = st.columns(4)
with col_a: status_card(1, "Export ONNX", "YOLO → .onnx via ultralytics")
with col_b: status_card(2, "Calib Data",  "Sample images → calib/ folder")
with col_c: status_card(3, "Parse HAR",   "hailomz parse → .har file")
with col_d: status_card(4, "Compile HEF", "hailomz compile → .hef file")

st.markdown("---")

# ==========================================
# STEP 1 — Export .pt → .onnx
# ==========================================
with st.expander("▸ Step 1 — Export .pt → .onnx", expanded=True):
    st.markdown('<div class="step-card"><div class="step-title">Export with YOLO CLI inside ai_env</div></div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        export_imgsz = st.number_input("Image size (imgsz)", value=512, step=32, key="exp_imgsz")
        opset        = st.number_input("ONNX opset", value=11, min_value=9, max_value=18, key="exp_opset")
    with c2:
        onnx_out_name = st.text_input("Output .onnx filename", value="model.onnx")
        os.makedirs(output_dir, exist_ok=True)
        onnx_full_path = os.path.join(output_dir, onnx_out_name)

    if st.button("▶ Run Export", key="btn_step1"):
        if not os.path.exists(pt_model_path):
            st.error(f"Cannot find model: {pt_model_path}")
        else:
            st.session_state.step_status[1] = "active"
            st.session_state.log_1 = ""
            with st.spinner("Exporting…"):
                env = activated_env(ai_env_path)
                # Disable GPU to avoid conflicts during export
                env["CUDA_VISIBLE_DEVICES"] = "-1"
                yolo_bin = os.path.join(ai_env_path, "bin", "yolo")
                cmd = [
                    yolo_bin,
                    "export",
                    f"model={pt_model_path}",
                    "format=onnx",
                    f"imgsz={export_imgsz}",
                    f"opset={opset}",
                ]
                rc, log = run_cmd(cmd, env=env)

            # YOLO saves the .onnx next to the .pt; move it to output_dir
            auto_onnx = pt_model_path.replace(".pt", ".onnx")
            if rc == 0 and os.path.exists(auto_onnx):
                shutil.move(auto_onnx, onnx_full_path)
                st.session_state.onnx_path = onnx_full_path
                st.session_state.step_status[1] = "done"
                log += f"\n✓ Moved to: {onnx_full_path}"
            else:
                st.session_state.step_status[1] = "error"
            st.session_state.log_1 = log
            st.rerun()

    show_log(1)
    if st.session_state.step_status[1] == "done":
        st.success(f"✓ ONNX saved: {st.session_state.onnx_path}")

# ==========================================
# STEP 2 — Generate calibration data
# ==========================================
with st.expander("▸ Step 2 — Generate Calibration Data", expanded=False):
    st.markdown('<div class="step-card"><div class="step-title">hailo_calibration_data.py from RasPi_YOLO repo</div></div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        calib_script  = st.text_input("Path to hailo_calibration_data.py",
                                       value=os.path.join(BASE_DIR, "RasPi_YOLO", "hailo_calibration_data.py"))
        data_dir      = st.text_input("Training images dir (--data_dir)",
                                       value=os.path.join(BASE_DIR, "train_data", "train", "images"))
    with c2:
        calib_out_dir = st.text_input("Calibration output dir (--target_dir)",
                                       value=os.path.join(output_dir, "calib"))
        calib_imgsz   = st.text_input("Image size (H W)", value="640 640")
        num_images    = st.number_input("Number of calibration images", value=256, min_value=10)
    if num_images < 200:
        st.warning("⚠️ Recommended minimum is 200 images. Below this, quantization may fail with NegativeSlopeExponentNonFixable errors.")

    if st.button("▶ Generate Calib Data", key="btn_step2"):
        if not os.path.exists(calib_script):
            st.error(f"Script not found: {calib_script}")
        elif not os.path.exists(data_dir):
            st.error(f"Images directory not found: {data_dir}")
        else:
            st.session_state.step_status[2] = "active"
            st.session_state.log_2 = ""
            with st.spinner("Generating calibration data…"):
                env = activated_env(ai_env_path)
                env["CUDA_VISIBLE_DEVICES"] = "-1"
                h, w = calib_imgsz.strip().split()
                cmd = [
                    python_in_venv(ai_env_path), calib_script,
                    "--data_dir",   data_dir,
                    "--target_dir", calib_out_dir,
                    "--image_size", h, w,
                    "--num_images", str(num_images),
                ]
                rc, log = run_cmd(cmd, env=env)

            if rc == 0:
                st.session_state.step_status[2] = "done"
            else:
                st.session_state.step_status[2] = "error"
            st.session_state.log_2 = log
            st.rerun()

    show_log(2)
    if st.session_state.step_status[2] == "done":
        st.success("✓ Calibration data generated.")

# ==========================================
# STEP 3 — hailomz parse → .har
# ==========================================
with st.expander("▸ Step 3 — Parse Model (hailomz parse → .har)", expanded=False):
    st.markdown('<div class="step-card"><div class="step-title">Runs inside hailo_dfc_env</div></div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        parse_onnx   = st.text_input("Input .onnx path", value=st.session_state.onnx_path or onnx_full_path)
        model_name   = st.selectbox("Base model name", ["yolov8n", "yolov8s", "yolov8m", "yolov8l"])
    with c2:
        har_out_name = st.text_input("Output .har filename", value=f"{model_name}.har")
        har_full_path = os.path.join(output_dir, har_out_name)

    st.markdown("**End-node names** (one per line — leave default for standard YOLOv8n)")
    default_end_nodes = (
        "/model.22/cv2.0/cv2.0.2/Conv\n"
        "/model.22/cv3.0/cv3.0.2/Conv\n"
        "/model.22/cv2.1/cv2.1.2/Conv\n"
        "/model.22/cv3.1/cv3.1.2/Conv\n"
        "/model.22/cv2.2/cv2.2.2/Conv\n"
        "/model.22/cv3.2/cv3.2.2/Conv"
    )
    end_nodes_raw = st.text_area("End node names", value=default_end_nodes, height=150)

    if st.button("▶ Run Parse", key="btn_step3"):
        if not os.path.exists(parse_onnx):
            st.error(f"ONNX not found: {parse_onnx}")
        else:
            st.session_state.step_status[3] = "active"
            st.session_state.log_3 = ""
            end_nodes = [n.strip() for n in end_nodes_raw.strip().splitlines() if n.strip()]
            with st.spinner("Parsing model — this may take a few minutes…"):
                env = activated_env(hailo_env_path)
                env["CUDA_VISIBLE_DEVICES"] = "-1"
                hailomz_bin = os.path.join(hailo_env_path, "bin", "hailomz")
                cmd = [
                    hailomz_bin, "parse", model_name,
                    "--ckpt",     parse_onnx,
                    "--hw-arch",  hw_arch,
                    "--end-node-names",
                ] + end_nodes
                rc, log = run_cmd(cmd, env=env)

            # hailomz saves the .har in cwd; move to output_dir
            local_har = f"{model_name}.har"
            if rc == 0:
                if os.path.exists(local_har):
                    shutil.move(local_har, har_full_path)
                    st.session_state.har_path = har_full_path
                    log += f"\n✓ Moved to: {har_full_path}"
                st.session_state.step_status[3] = "done"
            else:
                st.session_state.step_status[3] = "error"
            st.session_state.log_3 = log
            st.rerun()

    show_log(3)
    if st.session_state.step_status[3] == "done":
        st.success(f"✓ HAR saved: {st.session_state.har_path}")

# ==========================================
# STEP 4 — hailomz compile → .hef
# ==========================================
with st.expander("▸ Step 4 — Compile to .hef", expanded=False):
    st.markdown('<div class="step-card"><div class="step-title">Runs inside hailo_dfc_env — takes ~9 min</div></div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        compile_onnx = st.text_input("Input .onnx (for custom compile)",
                                      value=st.session_state.onnx_path or onnx_full_path,
                                      key="compile_onnx_in")
        calib_path_in = st.text_input("Calibration data path (--calib-path)",
                                       value=os.path.join(output_dir, "calib", "calib"))
    with c2:
        compile_model = st.selectbox("Base model name", ["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
                                      key="compile_model_sel")
        hef_out_name  = st.text_input("Output .hef filename", value="model.hef")
        hef_full_path = os.path.join(output_dir, hef_out_name)

    use_har = st.checkbox(
        "Use pre-parsed .har instead of .onnx (faster if Step 3 already ran)",
        value=bool(st.session_state.har_path)
    )
    if use_har:
        har_input = st.text_input("HAR file path", value=st.session_state.har_path)

    perf_profile = st.selectbox(
        "Performance profile",
        ["default", "fastest_single_control_flow", "balanced", "latency"],
        index=0,
        help="Use 'fastest_single_control_flow' if compilation fails with NegativeSlopeExponentNonFixable quantization errors — it bypasses the strict optimization that causes the issue on CPU."
    )

    # Count calib images so user knows what the compiler will actually see
    if os.path.exists(calib_path_in):
        calib_count = len([f for f in os.listdir(calib_path_in)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.npy'))])
        if calib_count < 200:
            st.warning(f"⚠️ Only {calib_count} calibration images found in {calib_path_in}. "
                       "The Hailo compiler recommends at least 1024 (minimum ~200). "
                       "Re-run Step 2 with a higher image count to avoid quantization failures.")
        else:
            st.success(f"✓ {calib_count} calibration images found — ready to compile.")

    st.info("⏱️ Compilation typically takes 8–10 minutes. The log will appear when complete.")

    if st.button("▶ Compile to .hef", type="primary", key="btn_step4"):
        if not os.path.exists(calib_path_in):
            st.error(f"Calibration path not found: {calib_path_in}")
        else:
            st.session_state.step_status[4] = "active"
            st.session_state.log_4 = ""
            with st.spinner("Compiling — grab a coffee, this takes ~9 minutes…"):
                env = activated_env(hailo_env_path)
                env["CUDA_VISIBLE_DEVICES"] = "-1"
                hailomz_bin = os.path.join(hailo_env_path, "bin", "hailomz")

                profile_args = ["--performance-profile", perf_profile] if perf_profile != "default" else []

                if use_har:
                    cmd = [
                        hailomz_bin, "compile", compile_model,
                        "--har",        har_input,
                        "--hw-arch",    hw_arch,
                        "--classes",    str(num_classes),
                        "--calib-path", calib_path_in,
                    ] + profile_args
                else:
                    cmd = [
                        hailomz_bin, "compile", compile_model,
                        "--ckpt",       compile_onnx,
                        "--hw-arch",    hw_arch,
                        "--classes",    str(num_classes),
                        "--calib-path", calib_path_in,
                    ] + profile_args
                rc, log = run_cmd(cmd, env=env)

            # hailomz writes .hef to cwd
            local_hef = f"{compile_model}.hef"
            if rc == 0:
                if os.path.exists(local_hef):
                    shutil.move(local_hef, hef_full_path)
                    st.session_state.hef_path = hef_full_path
                    log += f"\n✓ Moved to: {hef_full_path}"
                st.session_state.step_status[4] = "done"
            else:
                st.session_state.step_status[4] = "error"
            st.session_state.log_4 = log
            st.rerun()

    show_log(4)

    if st.session_state.step_status[4] == "done":
        st.success(f"✓ HEF ready: {st.session_state.hef_path}")
        st.balloons()
        if os.path.exists(st.session_state.hef_path):
            with open(st.session_state.hef_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download .hef",
                    data=f,
                    file_name=os.path.basename(st.session_state.hef_path),
                    mime="application/octet-stream",
                    type="primary",
                )

# ==========================================
# Sidebar — pipeline reset
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🧹 Reset Pipeline")
st.sidebar.caption("Clears all step statuses and logs without deleting output files.")
if st.sidebar.button("Reset All Steps"):
    for k in ["step_status", "log_1", "log_2", "log_3", "log_4", "onnx_path", "har_path", "hef_path"]:
        del st.session_state[k]
    st.rerun()

# ==========================================
# Sidebar — quick reference
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("📋 Quick Reference")
with st.sidebar.expander("Manual commands"):
    st.code("""# Export
yolo export model=pete.pt \\
  format=onnx imgsz=512 opset=11

# Calib data
python hailo_calibration_data.py \\
  --data_dir ./train/images/ \\
  --target_dir calib \\
  --image_size 640 640 \\
  --num_images 135

# Parse
hailomz parse yolov8n \\
  --ckpt model.onnx \\
  --hw-arch hailo10h \\
  --end-node-names ...

# Compile
hailomz compile yolov8n \\
  --ckpt model.onnx \\
  --hw-arch hailo10h \\
  --classes 4 \\
  --calib-path ./calib/calib/
""", language="bash")

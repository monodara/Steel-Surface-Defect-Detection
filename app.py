import streamlit as st
from ultralytics import YOLO # type: ignore
import numpy as np
from PIL import Image
import time

# --- Page Configuration ---
st.set_page_config(page_title="Steel Surface Defect Detection System", layout="wide")

st.title("🛡️ Steel Surface Defect Intelligent Detection System")
st.markdown("---")

# --- Sidebar Configuration ---
st.sidebar.header("📦 Model Configuration")
# Toggle between Engine (TensorRT) and PT (PyTorch)
model_path = st.sidebar.selectbox("Select Inference Engine", ["best.engine", "best.pt"])
conf_thres = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.356)
iou_thres = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45)

# Cache the model to avoid reloading on every interaction
@st.cache_resource
def load_model(path):
    return YOLO(path, task='detect')

try:
    model = load_model(model_path)
    st.sidebar.success(f"Successfully Loaded: {model_path}")
except Exception as e:
    st.sidebar.error(f"Failed to load model. Check file path.")

# --- Main UI Logic ---
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Upload Sample")
    uploaded_file = st.file_uploader("Supports JPG, PNG, JPEG", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Original Input Image", use_container_width=True)

with col2:
    st.header("🔍 Detection Results")
    if uploaded_file is not None:
        if st.button("🚀 Run Instant Inference"):
            # Record start time for performance benchmarking
            start_time = time.time()
            
            # Execute inference
            results = model.predict( # type: ignore
                source=input_image,  # type: ignore
                conf=conf_thres, 
                iou=iou_thres, 
                imgsz=800
            )[0]
            
            end_time = time.time()
            infer_time = (end_time - start_time) * 1000 # Convert to milliseconds
            
            # Render the annotated image
            res_plotted = results.plot()[:, :, ::-1] # BGR to RGB conversion
            st.image(res_plotted, caption="Defect Localization Map", use_container_width=True)
            
            # Performance Metrics
            st.subheader("📊 Performance Analysis")
            st.info(f"⏱️ **End-to-End Latency:** `{infer_time:.2f} ms` (Powered by RTX 5090)")
            
            # Defect Statistics
            st.subheader("📋 Detection Summary")
            detected_classes = results.boxes.cls.tolist() # type: ignore
            if not detected_classes:
                st.success("🎉 No defects detected. Quality Assurance Passed!")
            else:
                # Count occurrences of each defect type
                for c in set(detected_classes):
                    label = results.names[int(c)]
                    count = detected_classes.count(c)
                    st.warning(f"Detected **{label}**: {count} instance(s)")

# --- Raw Data Output (Optional) ---
if uploaded_file and 'results' in locals():
    st.markdown("---")
    with st.expander("See Raw Inference JSON"):
        st.json(results.tojson()) # type: ignore
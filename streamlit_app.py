import streamlit as st
from ultralytics import YOLO # type: ignore
import numpy as np
from PIL import Image
import time

# --- Configure page ---
st.set_page_config(
    page_title="Steel Surface Defect Detection AI", 
    page_icon="🔍",
    layout="centered" 
)

# CSS styles
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #2e7d32; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Steel Surface Defect Detection")
st.caption("Industrial QA System | Powered by YOLOv11 | Hosted on HF CPU")

# --- Load model - use cpu ---
@st.cache_resource
def load_hf_model():
    model = YOLO("best.pt", task='detect')
    return model

try:
    model = load_hf_model()
    st.sidebar.success("✅ Model: v5-Finetune (Ready)")
except Exception as e:
    st.sidebar.error("⚠️ Model 'best.pt' not found in root directory.")

# --- Sidebar: change confidence ---
st.sidebar.header("🛠️ Detection Settings")
conf_thres = st.sidebar.slider("Sensitivity (Confidence)", 0.0, 1.0, 0.35)

# --- UI ---
uploaded_file = st.file_uploader("Upload Surface Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Process uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    
    # Button to start inference
    if st.button("🚀 Start Intelligent Analysis"):
        with st.spinner("AI is examining the steel texture..."):
            # Record inference time
            t_start = time.time()
            
            # Inference with optimized settings for CPU
            results = model.predict( # type: ignore
                source=img,
                conf=conf_thres,
                imgsz=640,  
                device='cpu' 
            )[0]
            
            t_end = time.time()
            duration = (t_end - t_start) * 1000

            # Display results
            res_plotted = results.plot(labels=True, boxes=True)
            res_img = Image.fromarray(res_plotted[:, :, ::-1])
            st.image(res_img, caption="AI Detection Result", use_container_width=True)

            # Show metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Latency", f"{duration:.0f} ms")
            with col2:
                count = len(results.boxes) # type: ignore
                st.metric("Defects Found", count)

            # Defects
            if count > 0:
                st.subheader("📋 Defect Inventory")
                labels = results.boxes.cls.tolist() # type: ignore
                for c in set(labels):
                    name = results.names[int(c)]
                    num = labels.count(c)
                    st.write(f"📍 **{name}**: {num} instances")
            else:
                st.balloons()
                st.success("No surface defects detected. Part is within quality standards.")

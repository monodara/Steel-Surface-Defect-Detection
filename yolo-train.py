from ultralytics import YOLO
import os
import torch

if torch.backends.mps.is_available():
    device = "mps"
    print("🚀 Detect Apple Silicon, use MPS for acceleration")
elif torch.cuda.is_available():
    device = 0
    print("🚀 Detect NVIDIA GPU, use CUDA for acceleration")
else:
    device = "cpu"
    print("⚠️ Failed to detect GPU, use CPU for training (slower).")
# Initialise YOLO model
model = YOLO('yolov12.yaml', task='detect').load('yolo12m.pt')

# Display model information
model.info()

results = model.train(
    data='./steel_data/dataset.yaml',      
    epochs=100,
    
    batch=8,                 
    
    imgsz=640,
    patience=50,
    save=True,
    
    # ... other training parameters, refer to the documentation for more details
)

metrics = model.val()
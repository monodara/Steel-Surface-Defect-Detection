from ultralytics import YOLO # type: ignore

model = YOLO("yolo12m.pt")
model.info()

# v3
# v4
results = model.train(
    data='./steel_data/dataset.yaml',   
    project=f"/runs",      
    name="steel_detect_v4",          
    epochs=150,              
    batch=8,
    imgsz=800,
    patience=50,
    save=True,
    device='0',
    workers=8,
    cache=True,
    pretrained=True,
    optimizer='auto',
    verbose=True,
    overlap_mask=False,
    amp=True,

    hsv_h=0.0,            
    hsv_s=0.0,          
    hsv_v=0.3,            # Slightly reduce brightness augmentation to avoid overexposure

    close_mosaic=20,      

    # --- Generalization defense ---
    scale=0.9,             # Add scale diversity
    mixup=0.15,            # Moderately reduce mixup to prevent over-blurring
    weight_decay=0.003,    # Further strengthen regularization, strongly suppress test error
    label_smoothing=0.1,   # Restore to 0.1
    cos_lr=True,           # Enable cosine annealing for more precise convergence
    
    # --- Special: Hard sample(Crazing) tackling ---
    cls=1.8,               # Slightly reduce, prevent overfitting
    box=6.5,               # Continue fine-tuning localization weight, further tilt towards classification
)
# Evaluate the model
results = model.val()


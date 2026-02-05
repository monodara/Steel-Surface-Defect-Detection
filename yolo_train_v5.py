from ultralytics import YOLO # type: ignore

model = YOLO("yolo12m.pt")
model.info()

# v5
results = model.train(
    data='./steel_data/dataset.yaml',    
    project=f"/runs",       
    name="steel_detect_v5",            
    epochs=150,              
    batch=8,
    imgsz=800,
    patience=50,
    save=True,
    device='0',
    workers=4,
    cache=True,
    pretrained=True,
    optimizer='auto',
    verbose=True,
    overlap_mask=False,
    amp=True,

    # --- 1. Core features protection (Use logic of v3/v4 ) ---
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.4,

    # --- 2. Fusion strategy: v3's "will" + v4's "smoothness" ---
    cls=2.0,               # Go back to the v3 high classification weight, preserve F1 curve width
    box=7.0,               # Recover localization weight to v3 level, maintain box stability
    label_smoothing=0.05,  # Slightly reduce, make the model more "tough" in judging Crazing
    
    # --- 3. Generalization defense upgrade ---
    weight_decay=0.0025,   # Take the best of both, precisely filter noise
    cos_lr=True,           # Keep the cosine annealing, ensure precise convergence in later stages
    scale=0.8,             # Moderate scale augmentation
    mixup=0.15,            # Keep moderate fusion
    close_mosaic=25,       # 25 rounds enter fine-tuning, slightly more than v4
)
# Evaluate the model
results = model.val()


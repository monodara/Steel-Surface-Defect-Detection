from ultralytics import YOLO # type: ignore

model = YOLO("yolo12m.pt")
model.info()

# v3
results = model.train(
    data='./steel_data/dataset.yaml',    
    project=f"/runs",       
    name="steel_detect_v3",            
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

    # --- 1. Keep only grayscale features ---
    hsv_h=0.0,            # Close it explicitly, default is 0.015
    hsv_s=0.0,            # Close it explicitly, default is 0.7
    hsv_v=0.4,            # Keep brightness variation

    # --- 2. Improve classification and recall ---
    cls=2.0,              # Aggressive weight, solve Crazing misses
    box=7.0,              # Recovery localization weight, prevent overfitting (downgraded from 10.0)
    label_smoothing=0.15, # Increase smoothing, tolerate uncertainty in crack detection

    # --- 3. Augmentation logic: from hard paste to fusion ---
    copy_paste=0.1,       # Reduce harsh edges
    mixup=0.2,            # Enhance fusion sense, simulate crack texture

    # --- 4. Generalization defense ---
    weight_decay=0.002,   # Increase weight decay, filter out high-frequency noise
    close_mosaic=30       # Close mosaic early, let the model see "pure" steel plates
)
# Evaluate the model
results = model.val()


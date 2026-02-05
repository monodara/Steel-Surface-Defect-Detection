from ultralytics import YOLO # type: ignore

model = YOLO("yolo12m.pt")
model.info()

results = model.train(
    data='./steel_data/dataset.yaml',    
    project=f"runs",    
    name="steel_refined_v2",             # name of experiment
    epochs=150,
    batch=12,
    imgsz=800,
    patience=50,
    save=True,
    device='0',
    workers=4,
    cache=True,
    pretrained=True,
    optimizer='auto',
    verbose=True,
    overlap_mask=True,
    weight_decay=0.001,

    scale=0.6,
    mixup=0.1,
    mosaic=1.0,         
    copy_paste=0.3,       # Increased from 0.1 to 0.3, adding rare defects occurring together

    # --- Training Logic Optimization ---
    close_mosaic=20,      # Fine-tune longer on clean images to stabilize background judgment
    warmup_epochs=3.0,

    # --- Loss Weight Adjustment ---
    box=10.0,             # Keep high box loss weight to ensure precise localization
    cls=1.5,              # Increase classification loss weight (default is 0.5) to address "class imbalance"
    label_smoothing=0.1,  # Add label smoothing to prevent model from being overconfident about background noise
)
# Evaluate the model
results = model.val()


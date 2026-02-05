from ultralytics import YOLO # type: ignore

model = YOLO("yolo12m.pt")
model.info()

# v5-finetune
results = model.train(
    data='./steel_data/dataset.yaml',    
    project=f"/runs",       
    name="steel_detect_v5_finetune", 
    epochs=50,          
    batch=8,
    imgsz=800,
    device=0,
    
    # --- Core finetune arguments ---
    lr0=0.0001,            # lower initial learning rate
    lrf=0.01,              # final learning rate factor
    warmup_epochs=0,       # skip warmup for finetuning
    
    # High classification weights to focus on difficult classes
    cls=2.5,               
    
    # Freeze Backbone layers to retain learned features
    freeze=10,  
    
    # --- Reduce interfering augmentation ---
    mixup=0.0,            
    copy_paste=0.0,
    mosaic=1.0,         
    
    weight_decay=0.0025,
    cos_lr=True
)


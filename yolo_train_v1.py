from ultralytics import YOLO # type: ignore

model = YOLO("yolo12m.pt")
model.info()

results = model.train(
    data='./steel_data/dataset.yaml',      # path to dataset configuration file
    epochs=100,               # training times
    batch=8,                 
    imgsz=640,               
    patience=50,             
    save=True,               
    device='0',              
    workers=4,               
    pretrained=True,        
    optimizer='auto',        
    verbose=True,        

    # parameters for augmentation
    scale=0.5,              # Random scale augmentation
    mosaic=1.0,             # Mosaic augmentation probability - combination of 4 images
    mixup=0.1,              # Mixup augmentation probability - linear combination of 2 images
    copy_paste=0.1,         # Copy-paste augmentation probability - paste objects from one image to another
)
# Evaluate the model
results = model.val()


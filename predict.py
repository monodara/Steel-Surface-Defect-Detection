import os
import pandas as pd
from ultralytics import YOLO

model = YOLO('best.pt')

# Set paths and prepare directories
test_image_dir = 'steel_data/test/images'
save_dir = 'runs/detect/predict'
os.makedirs(save_dir, exist_ok=True)

all_test_images = sorted([f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.png'))])

csv_lines = ["image_id,bbox,category_id,confidence"]

print(f"Processing {len(all_test_images)} images ...")

for img_name in all_test_images:
    img_id = os.path.splitext(img_name)[0]
    results = model(os.path.join(test_image_dir, img_name), conf=0.1)
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        csv_lines.append(f"{img_id},[0, 0, 0, 0],0,0")
    else:
        for box in boxes:
            c = box.xyxy[0].tolist()
            bbox_str = f"[{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}, {c[3]:.2f}]"
            cls_id = int(box.cls[0])
            conf = round(float(box.conf[0]), 4)
            csv_lines.append(f"{img_id},{bbox_str},{cls_id},{conf}")

with open('submission.csv', 'w', encoding='utf-8') as f:
    f.write("\n".join(csv_lines))

print("✅ submission.csv Generated Successfully!")
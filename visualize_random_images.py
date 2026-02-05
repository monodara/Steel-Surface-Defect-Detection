import cv2
import os
import random
from glob import glob
import xml.etree.ElementTree as ET

image_dir = "/Users/gawin/Documents/projects/steel-defect-detection/steel_data/train/IMAGES"
annotation_dir = "/Users/gawin/Documents/projects/steel-defect-detection/steel_data/train/ANNOTATIONS"

classes = ['crazing', 'inclusion', 'pitted_surface', 'scratches', 'patches', 'rolled-in_scale']

image_files = glob(os.path.join(image_dir, "*.jpg"))
selected_images = random.sample(image_files, 2)

combined_img_list = []
for idx, image_path in enumerate(selected_images):
    base_name = os.path.basename(image_path).split('.')[0]
    annotation_path = os.path.join(annotation_dir, base_name + ".xml")
    
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # 解析XML标注
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    for obj in root.findall('object'):
        # 获取缺陷类别
        name_elem = obj.find('n')
        if name_elem is None:
            name_elem = obj.find('name')
        class_name = name_elem.text.strip()
        class_id = classes.index(class_name)
        
        box = obj.find('bndbox')
        xmin = int(float(box.find('xmin').text))
        ymin = int(float(box.find('ymin').text))
        xmax = int(float(box.find('xmax').text))
        ymax = int(float(box.find('ymax').text))
        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        label_y = ymin - 5 if ymin > 15 else ymax + 15
        cv2.putText(img, class_name, (xmin, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    title = f"{os.path.basename(image_path)}"
    img_with_title = cv2.copyMakeBorder(img, 0, 30, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    cv2.putText(img_with_title, title, (10, img.shape[0]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    combined_img_list.append(img_with_title)

final_img = cv2.hconcat(combined_img_list)

cv2.imwrite("visualized_defects.jpg", final_img)

cv2.imshow("Steel Defect Detection - Random Images", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
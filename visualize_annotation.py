import cv2
import os
from glob import glob

# 读取图片和标注文件
image_path = "/Users/gawin/Documents/projects/steel-defect-detection/steel_data/train/IMAGES/101.jpg"
annotation_path = "/Users/gawin/Documents/projects/steel-defect-detection/steel_data/train/ANNOTATIONS/101.xml"

# 定义缺陷类别
classes = ['crazing', 'inclusion', 'pitted_surface', 'scratches', 'patches', 'rolled-in_scale']

# 加载图片
img = cv2.imread(image_path)
h, w = img.shape[:2]

# 解析XML标注
import xml.etree.ElementTree as ET
tree = ET.parse(annotation_path)
root = tree.getroot()

for obj in root.findall('object'):
    # 获取缺陷类别
    name_elem = obj.find('n')
    if name_elem is None:
        name_elem = obj.find('name')
    class_name = name_elem.text.strip()
    class_id = classes.index(class_name)
    
    # 获取边界框坐标
    box = obj.find('bndbox')
    xmin = int(float(box.find('xmin').text))
    ymin = int(float(box.find('ymin').text))
    xmax = int(float(box.find('xmax').text))
    ymax = int(float(box.find('ymax').text))
    
    # 在图片上绘制边界框
    color = (0, 255, 255)  # Yellow color for all defect types
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(img, class_name, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 显示图片
cv2.imshow("Steel Defect Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
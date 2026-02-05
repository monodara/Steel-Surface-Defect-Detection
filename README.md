# 🛡️ Steel Surface Defect Detection with YOLOv12

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/Model-YOLOv12m-green.svg)](https://github.com/ultralytics/ultralytics)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](YOUR_HUGGINGFACE_SPACE_URL_HERE)


An end-to-end deep learning pipeline for industrial steel quality control. This repository contains the complete training configuration, evaluation metrics, and the source code for the live deployment.

---

## 🌐 Live Demo
The model is deployed as an interactive web application on **Hugging Face Spaces**. You can test the defect detection system online without any local setup:

👉 **[Launch Steel Detection App on Hugging Face](https://monodara-steel-defect-detection.hf.space)**

---

## 📊 Training Results & Evaluation

The model was fine-tuned over 5 major iterations (v1-v5) to achieve the best balance between precision and recall on textured steel surfaces.

### 📈 Performance Metrics
| Metric | Visualization | Description |
| :--- | :--- | :--- |
| **Training Summary** | `results.png` | Overview of loss convergence and mAP improvement over epochs. |
| **F1-Curve** | `F1_curve.png` | Shows the optimal confidence threshold for maximizing F1-score. |
| **Confusion Matrix** | `confusion_matrix.png` | Detailed breakdown of per-class classification accuracy vs. background. |



### 🧪 Experiment Configuration
- **Base Model:** YOLOv12m
- **Input Size:** 640px
- **Augmentation:** Scale, Mixup, Mosaic, Copy_paste, etc.
- **Optimizer:** SGD / AdamW (v5 Final)

    The experiment process is shared in [this Medium article](https://medium.com/@monodara.lu/steel-defect-detection-from-theory-to-practice-d275d1ba206f).


---


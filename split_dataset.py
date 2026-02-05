import os
import random
from sklearn.model_selection import train_test_split
import glob

def split_images_to_txt():
    # 设置随机种子，确保每次分割结果相同
    random.seed(42)
    
    # 定义路径
    images_dir = "/Users/gawin/Downloads/22-Pytorch与视觉检测/yolo-cases/steel_data/train/images"
    output_dir = "/Users/gawin/Downloads/22-Pytorch与视觉检测/yolo-cases/steel_data/train"
    
    # 获取所有jpg文件
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    
    # 提取文件名（不含路径）
    image_names = [os.path.basename(img) for img in image_files]
    
    # 打印文件总数
    total_count = len(image_names)
    print(f"Total images found: {total_count}")
    
    # 首先划分出训练集（70%）和临时集（30%，后续再分为验证集和测试集）
    train_files, temp_files = train_test_split(
        image_names, 
        test_size=0.3, 
        random_state=42
    )
    
    # 将临时集按比例划分为验证集（占总数20%）和测试集（占总数10%）
    # 验证集占剩余30%中的2/3（即总数据的20%）
    validation_files, test_files = train_test_split(
        temp_files, 
        test_size=0.33,  # 约等于1/3，使得验证集约20%，测试集约10%
        random_state=42
    )
    
    # 输出统计信息
    print(f"Train set: {len(train_files)} files ({len(train_files)/total_count*100:.1f}%)")
    print(f"Validation set: {len(validation_files)} files ({len(validation_files)/total_count*100:.1f}%)")
    print(f"Test set: {len(test_files)} files ({len(test_files)/total_count*100:.1f}%)")
    
    # 写入训练集文件
    train_output_path = os.path.join(output_dir, "train_set.txt")
    with open(train_output_path, 'w') as f:
        for img_name in train_files:
            f.write(f"./images/{img_name}\n")
    
    # 写入验证集文件
    val_output_path = os.path.join(output_dir, "validation_set.txt")
    with open(val_output_path, 'w') as f:
        for img_name in validation_files:
            f.write(f"./images/{img_name}\n")
    
    # 写入测试集文件
    test_output_path = os.path.join(output_dir, "test_set.txt")
    with open(test_output_path, 'w') as f:
        for img_name in test_files:
            f.write(f"./images/{img_name}\n")
    
    print(f"Files saved:")
    print(f"- Train set: {train_output_path}")
    print(f"- Validation set: {val_output_path}")
    print(f"- Test set: {test_output_path}")

if __name__ == "__main__":
    split_images_to_txt()
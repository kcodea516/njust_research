"""
数据集下载与准备脚本
=====================
功能：
1. 下载公开车辆数据集 (BIT-Vehicle, COCO车辆子集等)
2. 格式转换为 YOLO 格式
3. 数据集划分
4. 生成 dataset.yaml 配置文件

使用方式:
    # 使用 COCO 数据集中的车辆类别作为训练数据（推荐，最简单）
    python src/download_dataset.py --dataset coco-vehicle
    
    # 使用 BIT-Vehicle 数据集
    python src/download_dataset.py --dataset bit-vehicle
    
    # 使用已有的自定义数据集
    python src/download_dataset.py --dataset custom --source-dir /path/to/your/data
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

import yaml
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# COCO 数据集中车辆相关类别的ID和映射
COCO_VEHICLE_CLASSES = {
    2: 0,   # car -> car
    5: 1,   # bus -> bus
    7: 2,   # truck -> truck
}

COCO_CLASS_NAMES = {
    0: "car",
    1: "bus",
    2: "truck",
}


def download_coco_vehicle(output_dir: str, year: str = "2017"):
    """
    下载 COCO 数据集中的车辆类别子集
    
    这是最推荐的方式：
    - COCO 是计算机视觉领域最广泛使用的数据集之一
    - 包含大量真实场景中的车辆图片
    - 标注质量高
    - YOLOv8 原生就支持 COCO 格式
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  下载 COCO 车辆子集")
    print("=" * 60)
    print(f"  输出目录: {output_dir}")
    print(f"  车辆类别: car, bus, truck")
    print()
    
    # 直接使用 ultralytics 内置的 COCO 数据集支持
    # 这会自动下载并缓存 COCO 数据集
    print("💡 方案说明:")
    print("   YOLOv8 可以直接使用 COCO 数据集进行训练。")
    print("   推荐使用以下命令直接开始训练：")
    print()
    print("   python src/train.py --data coco.yaml --model yolov8s.pt --epochs 50")
    print()
    print("   YOLOv8 会自动下载 COCO 数据集。")
    print("   如果你只想训练车辆类别，使用下面的自定义配置：")
    print()
    
    # Create a COCO vehicle-only config
    coco_vehicle_config = {
        "path": str(output_path.resolve()),
        "train": "images/train2017",
        "val": "images/val2017",
        "nc": 3,
        "names": COCO_CLASS_NAMES,
        "download": f"""
import subprocess
from pathlib import Path

# Download COCO 2017 images
data_dir = Path('{str(output_path.resolve())}')
data_dir.mkdir(parents=True, exist_ok=True)

urls = [
    'http://images.cocodataset.org/zips/train2017.zip',
    'http://images.cocodataset.org/zips/val2017.zip',
    'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
]

print('请手动下载以下文件并解压到数据目录:')
for url in urls:
    print(f'  {{url}}')
print(f'解压到: {{data_dir}}')
"""
    }
    
    config_path = output_path / "coco_vehicle.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(coco_vehicle_config, f, default_flow_style=False, allow_unicode=True)
    
    # Create a simpler alternative: download script for small demo
    create_demo_dataset(str(output_path / "demo"))
    
    print(f"\n✅ COCO 车辆配置文件已创建: {config_path}")
    print(f"✅ 演示数据集已创建: {output_path / 'demo'}")
    print()
    print("📋 后续步骤:")
    print("  1. 如需完整 COCO 数据集 (约20GB):")
    print("     wget http://images.cocodataset.org/zips/train2017.zip")
    print("     wget http://images.cocodataset.org/zips/val2017.zip")
    print("     wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    print()
    print("  2. 快速测试（使用演示数据集）:")
    print(f"     python src/train.py --data {output_path / 'demo' / 'dataset.yaml'} --epochs 5")
    

def create_demo_dataset(output_dir: str, num_images: int = 50):
    """
    创建一个小型演示数据集，用于验证代码流程是否正确
    
    生成合成的车辆图片（彩色矩形 + 标签），
    不用于实际训练，仅用于测试代码流程。
    """
    output_path = Path(output_dir)
    
    # Create directory structure
    for split in ["train", "val"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    print(f"🎨 创建演示数据集 ({num_images} 张合成图片)...")
    
    class_names = {0: "car", 1: "bus", 2: "truck"}
    class_colors = {
        0: (60, 60, 200),    # Red for car
        1: (200, 150, 50),   # Blue for bus
        2: (50, 200, 50),    # Green for truck
    }
    
    np.random.seed(42)
    
    for i in range(num_images):
        # Determine split
        split = "train" if i < int(num_images * 0.8) else "val"
        
        # Create synthetic image (640x480)
        img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        
        # Add 1-3 "vehicles" (colored rectangles)
        num_objects = np.random.randint(1, 4)
        labels = []
        
        for _ in range(num_objects):
            cls_id = np.random.randint(0, 3)
            
            # Random bounding box
            w = np.random.randint(60, 200)
            h = np.random.randint(40, 150)
            x1 = np.random.randint(0, 640 - w)
            y1 = np.random.randint(0, 480 - h)
            x2 = x1 + w
            y2 = y1 + h
            
            # Draw rectangle
            color = class_colors[cls_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
            
            # Add text label
            label_text = class_names[cls_id]
            cv2.putText(img, label_text, (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # YOLO format: class cx cy w h (normalized)
            cx = (x1 + x2) / 2 / 640
            cy = (y1 + y2) / 2 / 480
            nw = w / 640
            nh = h / 480
            labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        
        # Save image
        img_path = output_path / "images" / split / f"demo_{i:04d}.jpg"
        cv2.imwrite(str(img_path), img)
        
        # Save label
        lbl_path = output_path / "labels" / split / f"demo_{i:04d}.txt"
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(labels))
    
    # Create dataset.yaml
    config = {
        "path": str(output_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": class_names,
    }
    
    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    train_count = int(num_images * 0.8)
    val_count = num_images - train_count
    print(f"✅ 演示数据集已创建:")
    print(f"   训练集: {train_count} 张")
    print(f"   验证集: {val_count} 张")
    print(f"   配置文件: {yaml_path}")


def setup_bit_vehicle(output_dir: str):
    """
    BIT-Vehicle 数据集使用说明
    
    BIT-Vehicle 包含 9,850 张车辆图像，分为 6 类：
    Bus, Microbus, Minivan, Sedan, SUV, Truck
    """
    print("=" * 60)
    print("  BIT-Vehicle 数据集")
    print("=" * 60)
    print()
    print("📋 BIT-Vehicle 数据集信息:")
    print("   图片数量: 9,850")
    print("   类别: Bus, Microbus, Minivan, Sedan, SUV, Truck")
    print()
    print("📥 下载方式:")
    print("   1. 访问: https://iitlab.bit.edu.cn/mcislab/vehicledb/")
    print("   2. 或搜索 'BIT-Vehicle Dataset' ")
    print("   3. 下载后解压到目标目录")
    print()
    print(f"📂 请将数据解压到: {output_dir}")
    print()
    print("📋 解压后需要的目录结构:")
    print(f"   {output_dir}/")
    print("   ├── Bus/")
    print("   ├── Microbus/")
    print("   ├── Minivan/")
    print("   ├── Sedan/")
    print("   ├── SUV/")
    print("   └── Truck/")
    print()
    print("💡 然后运行以下命令进行格式转换:")
    print(f"   python src/download_dataset.py --dataset custom --source-dir {output_dir}")


def convert_classification_to_detection(
    source_dir: str,
    output_dir: str,
    class_mapping: dict = None
):
    """
    将分类数据集（每个类别一个文件夹）转换为 YOLO 检测格式
    
    适用于 BIT-Vehicle 等分类数据集
    """
    import cv2
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if class_mapping is None:
        # Auto-detect classes from subdirectories
        subdirs = sorted([d for d in source_path.iterdir() if d.is_dir()])
        class_mapping = {d.name: i for i, d in enumerate(subdirs)}
    
    print(f"📋 检测到 {len(class_mapping)} 个类别:")
    for name, idx in class_mapping.items():
        print(f"   [{idx}] {name}")
    
    # Collect all images
    all_images = []
    for class_name, class_id in class_mapping.items():
        class_dir = source_path / class_name
        if not class_dir.exists():
            print(f"⚠️  目录不存在: {class_dir}")
            continue
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            for img_file in class_dir.glob(ext):
                all_images.append((img_file, class_id))
    
    if not all_images:
        print("❌ 未找到任何图片！")
        return
    
    print(f"\n📊 共找到 {len(all_images)} 张图片")
    
    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(all_images)
    
    n = len(all_images)
    n_train = int(n * 0.8)
    n_val = int(n * 0.15)
    
    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:]
    }
    
    for split_name, images in splits.items():
        img_dir = output_path / "images" / split_name
        lbl_dir = output_path / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file, class_id in images:
            # Copy image
            dst_img = img_dir / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # Create YOLO label (full image bbox since it's a classification dataset)
            # For classification datasets, we treat the entire image as one bounding box
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w = img.shape[:2]
                # Full image bounding box
                cx, cy = 0.5, 0.5
                nw, nh = 0.95, 0.95  # Slightly smaller than full image
                
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                with open(lbl_file, 'w') as f:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
    
    # Create dataset.yaml
    names = {v: k for k, v in class_mapping.items()}
    config = {
        "path": str(output_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_mapping),
        "names": names,
    }
    
    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n✅ 数据集转换完成:")
    for split_name, images in splits.items():
        print(f"   {split_name}: {len(images)} 张")
    print(f"   配置文件: {yaml_path}")


try:
    import cv2
except ImportError:
    print("⚠️ OpenCV 未安装，部分功能不可用")
    print("   请运行: pip install opencv-python")


def main():
    parser = argparse.ArgumentParser(
        description="超限车辆智能感知系统 - 数据集下载与准备"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["coco-vehicle", "bit-vehicle", "custom", "demo"],
        help="数据集类型"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="输出目录 (默认: data/processed)"
    )
    parser.add_argument(
        "--source-dir", default=None,
        help="原始数据目录 (仅 custom 模式需要)"
    )
    parser.add_argument(
        "--num-images", type=int, default=50,
        help="演示数据集图片数量 (仅 demo 模式)"
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "data" / "processed")
    
    if args.dataset == "coco-vehicle":
        download_coco_vehicle(args.output_dir)
    elif args.dataset == "bit-vehicle":
        setup_bit_vehicle(args.output_dir)
    elif args.dataset == "demo":
        create_demo_dataset(args.output_dir, args.num_images)
    elif args.dataset == "custom":
        if args.source_dir is None:
            print("❌ 自定义数据集需要指定 --source-dir 参数")
            sys.exit(1)
        convert_classification_to_detection(args.source_dir, args.output_dir)
    

if __name__ == "__main__":
    main()

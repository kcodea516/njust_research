"""
数据预处理模块
===============
功能：
1. 数据集格式验证与转换
2. 图像增强（亮度/对比度/天气模拟等）
3. 数据集划分（train/val/test）
4. 数据统计与可视化
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# ============================================================
# 车辆分类类别定义
# ============================================================
# 根据项目需求：货车、客车、危化品运输车等
VEHICLE_CLASSES = {
    0: "car",           # 小轿车
    1: "bus",           # 客车
    2: "truck",         # 货车
    3: "van",           # 面包车/厢式货车
    4: "suv",           # SUV
    5: "tanker",        # 危化品运输车/罐车
}

CLASS_NAMES_CN = {
    0: "小轿车",
    1: "客车",
    2: "货车",
    3: "面包车",
    4: "SUV",
    5: "危化品运输车",
}


def validate_yolo_dataset(data_dir: str) -> dict:
    """
    验证 YOLO 格式数据集的完整性
    
    YOLO格式要求:
    data_dir/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/ (可选)
    └── labels/
        ├── train/
        ├── val/
        └── test/ (可选)
    """
    data_path = Path(data_dir)
    report = {
        "valid": True,
        "errors": [],
        "stats": {}
    }
    
    for split in ["train", "val"]:
        img_dir = data_path / "images" / split
        lbl_dir = data_path / "labels" / split
        
        if not img_dir.exists():
            report["valid"] = False
            report["errors"].append(f"缺少目录: {img_dir}")
            continue
        
        if not lbl_dir.exists():
            report["valid"] = False
            report["errors"].append(f"缺少目录: {lbl_dir}")
            continue
        
        # Count images and labels
        img_files = set()
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            img_files.update(f.stem for f in img_dir.glob(ext))
        
        lbl_files = set(f.stem for f in lbl_dir.glob('*.txt'))
        
        # Check matching
        imgs_without_labels = img_files - lbl_files
        labels_without_imgs = lbl_files - img_files
        
        if imgs_without_labels:
            report["errors"].append(
                f"{split}: {len(imgs_without_labels)} 张图片缺少标签文件"
            )
        
        if labels_without_imgs:
            report["errors"].append(
                f"{split}: {len(labels_without_imgs)} 个标签文件缺少对应图片"
            )
        
        # Class distribution
        class_counts = Counter()
        total_boxes = 0
        for lbl_file in lbl_dir.glob('*.txt'):
            with open(lbl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        total_boxes += 1
        
        report["stats"][split] = {
            "num_images": len(img_files),
            "num_labels": len(lbl_files),
            "total_boxes": total_boxes,
            "class_distribution": dict(class_counts)
        }
    
    return report


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42
):
    """
    将原始数据集按比例划分为 train/val/test
    
    Args:
        source_dir: 原始数据目录（包含 images/ 和 labels/ 子目录）
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "数据集划分比例之和必须为1"
    
    random.seed(seed)
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Collect all image files
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        if (source_path / "images").exists():
            img_files.extend((source_path / "images").glob(ext))
        else:
            img_files.extend(source_path.glob(ext))
    
    if not img_files:
        print("❌ 未找到任何图片文件！")
        return
    
    # Shuffle and split
    random.shuffle(img_files)
    n = len(img_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        "train": img_files[:n_train],
        "val": img_files[n_train:n_train + n_val],
        "test": img_files[n_train + n_val:]
    }
    
    print(f"📊 数据集划分: train={len(splits['train'])}, "
          f"val={len(splits['val'])}, test={len(splits['test'])}")
    
    for split_name, files in splits.items():
        if not files:
            continue
        
        img_out = output_path / "images" / split_name
        lbl_out = output_path / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        
        for img_file in tqdm(files, desc=f"处理 {split_name}"):
            # Copy image
            shutil.copy2(img_file, img_out / img_file.name)
            
            # Copy corresponding label
            lbl_file = img_file.parent.parent / "labels" / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                lbl_file = img_file.with_suffix('.txt')
            if lbl_file.exists():
                shutil.copy2(lbl_file, lbl_out / f"{img_file.stem}.txt")
    
    print(f"✅ 数据集划分完成，保存至: {output_path}")


def augment_image(image: np.ndarray, mode: str = "random") -> np.ndarray:
    """
    数据增强函数
    
    Args:
        image: 输入图像 (BGR)
        mode: 增强模式
            - "brightness": 随机亮度调整
            - "contrast": 随机对比度调整
            - "blur": 高斯模糊（模拟雾天）
            - "noise": 添加高斯噪声
            - "rain": 模拟雨天效果
            - "night": 模拟夜间效果
            - "random": 随机组合以上效果
    
    Returns:
        增强后的图像
    """
    result = image.copy()
    
    if mode == "random":
        modes = random.sample(
            ["brightness", "contrast", "blur", "noise"],
            k=random.randint(1, 3)
        )
    else:
        modes = [mode]
    
    for m in modes:
        if m == "brightness":
            # 随机亮度调整 (0.5~1.5倍)
            factor = random.uniform(0.5, 1.5)
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        elif m == "contrast":
            # 随机对比度调整
            factor = random.uniform(0.5, 1.5)
            mean = np.mean(result, axis=(0, 1), keepdims=True)
            result = np.clip((result - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        elif m == "blur":
            # 高斯模糊（模拟雾天/远距离）
            ksize = random.choice([3, 5, 7])
            result = cv2.GaussianBlur(result, (ksize, ksize), 0)
        
        elif m == "noise":
            # 高斯噪声（模拟低光照）
            noise = np.random.normal(0, random.uniform(5, 25), result.shape)
            result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        elif m == "rain":
            # 简单雨天模拟
            rain = np.zeros_like(result)
            for _ in range(random.randint(100, 500)):
                x = random.randint(0, result.shape[1] - 1)
                y = random.randint(0, result.shape[0] - 1)
                length = random.randint(10, 30)
                cv2.line(rain, (x, y), (x + random.randint(-2, 2), y + length),
                        (200, 200, 200), 1)
            result = cv2.addWeighted(result, 0.85, rain, 0.15, 0)
        
        elif m == "night":
            # 夜间效果
            result = (result * 0.3).astype(np.uint8)
            noise = np.random.normal(0, 10, result.shape)
            result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return result


def batch_augment(
    input_dir: str,
    output_dir: str,
    augment_per_image: int = 3,
    modes: list = None
):
    """
    批量数据增强
    
    Args:
        input_dir: 输入图片目录
        output_dir: 输出目录
        augment_per_image: 每张图片生成的增强版本数
        modes: 增强模式列表
    """
    if modes is None:
        modes = ["brightness", "contrast", "blur", "noise", "rain", "night"]
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        img_files.extend(input_path.glob(ext))
    
    print(f"🖼️  找到 {len(img_files)} 张图片，每张生成 {augment_per_image} 个增强版本")
    
    for img_file in tqdm(img_files, desc="数据增强"):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        # Copy original
        shutil.copy2(img_file, output_path / img_file.name)
        
        # Generate augmented versions
        for i in range(augment_per_image):
            mode = random.choice(modes)
            aug_img = augment_image(img, mode)
            aug_name = f"{img_file.stem}_aug{i}_{mode}{img_file.suffix}"
            cv2.imwrite(str(output_path / aug_name), aug_img)
    
    total = len(img_files) * (1 + augment_per_image)
    print(f"✅ 数据增强完成，共 {total} 张图片，保存至: {output_path}")


def visualize_dataset_stats(data_dir: str, save_path: str = None):
    """
    可视化数据集统计信息
    
    生成:
    1. 各类别数量分布柱状图
    2. 训练集/验证集对比图
    3. 边界框尺寸分布
    """
    report = validate_yolo_dataset(data_dir)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("数据集统计信息", fontsize=16)
    
    # 1. Class distribution (all splits combined)
    all_class_counts = Counter()
    for split, stats in report["stats"].items():
        for cls_id, count in stats.get("class_distribution", {}).items():
            all_class_counts[cls_id] += count
    
    if all_class_counts:
        classes = sorted(all_class_counts.keys())
        counts = [all_class_counts[c] for c in classes]
        labels = [VEHICLE_CLASSES.get(c, f"class_{c}") for c in classes]
        
        bars = axes[0].bar(labels, counts, color=plt.cm.Set3(np.linspace(0, 1, len(classes))))
        axes[0].set_title("各类别样本数量")
        axes[0].set_xlabel("车辆类别")
        axes[0].set_ylabel("数量")
        for bar, count in zip(bars, counts):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        str(count), ha='center', va='bottom', fontsize=9)
    
    # 2. Split distribution
    split_names = list(report["stats"].keys())
    split_counts = [report["stats"][s]["num_images"] for s in split_names]
    
    axes[1].pie(split_counts, labels=split_names, autopct='%1.1f%%',
                colors=['#66b3ff', '#ff9999', '#99ff99'])
    axes[1].set_title("数据集划分比例")
    
    # 3. Image count per split
    axes[2].bar(split_names, split_counts,
                color=['#4CAF50', '#FF9800', '#2196F3'])
    axes[2].set_title("各集合图片数量")
    axes[2].set_ylabel("图片数量")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 统计图已保存至: {save_path}")
    else:
        save_path = str(Path(data_dir) / "dataset_stats.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 统计图已保存至: {save_path}")
    
    plt.close()
    return report


def create_dataset_yaml(
    data_dir: str,
    output_path: str = None,
    num_classes: int = None,
    class_names: dict = None
):
    """
    创建 YOLOv8 数据集配置文件 (dataset.yaml)
    
    Args:
        data_dir: 数据集根目录
        output_path: 输出文件路径
        num_classes: 类别数量
        class_names: 类别名称字典
    """
    if class_names is None:
        class_names = VEHICLE_CLASSES
    if num_classes is None:
        num_classes = len(class_names)
    
    data_path = Path(data_dir).resolve()
    
    config = {
        "path": str(data_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": num_classes,
        "names": class_names
    }
    
    if output_path is None:
        output_path = str(data_path / "dataset.yaml")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 数据集配置文件已创建: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="超限车辆智能感知系统 - 数据预处理工具"
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # validate command
    validate_parser = subparsers.add_parser("validate", help="验证数据集完整性")
    validate_parser.add_argument("--data-dir", required=True, help="数据集目录")
    
    # split command
    split_parser = subparsers.add_parser("split", help="划分数据集")
    split_parser.add_argument("--source-dir", required=True, help="原始数据目录")
    split_parser.add_argument("--output-dir", required=True, help="输出目录")
    split_parser.add_argument("--train-ratio", type=float, default=0.8)
    split_parser.add_argument("--val-ratio", type=float, default=0.15)
    split_parser.add_argument("--test-ratio", type=float, default=0.05)
    split_parser.add_argument("--seed", type=int, default=42)
    
    # augment command
    aug_parser = subparsers.add_parser("augment", help="批量数据增强")
    aug_parser.add_argument("--input-dir", required=True, help="输入图片目录")
    aug_parser.add_argument("--output-dir", required=True, help="输出目录")
    aug_parser.add_argument("--num-augment", type=int, default=3,
                           help="每张图片生成的增强版本数")
    
    # stats command
    stats_parser = subparsers.add_parser("stats", help="数据集统计与可视化")
    stats_parser.add_argument("--data-dir", required=True, help="数据集目录")
    stats_parser.add_argument("--save-path", default=None, help="统计图保存路径")
    
    # yaml command
    yaml_parser = subparsers.add_parser("yaml", help="创建数据集配置文件")
    yaml_parser.add_argument("--data-dir", required=True, help="数据集目录")
    yaml_parser.add_argument("--output", default=None, help="输出文件路径")
    
    args = parser.parse_args()
    
    if args.command == "validate":
        report = validate_yolo_dataset(args.data_dir)
        print("\n📋 数据集验证报告:")
        print(f"  有效性: {'✅ 有效' if report['valid'] else '❌ 存在问题'}")
        if report["errors"]:
            print("  问题:")
            for err in report["errors"]:
                print(f"    ⚠️  {err}")
        for split, stats in report["stats"].items():
            print(f"\n  [{split}]")
            print(f"    图片数: {stats['num_images']}")
            print(f"    标签数: {stats['num_labels']}")
            print(f"    标注框总数: {stats['total_boxes']}")
            if stats["class_distribution"]:
                print("    类别分布:")
                for cls_id, count in sorted(stats["class_distribution"].items()):
                    name = VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")
                    print(f"      {name}: {count}")
    
    elif args.command == "split":
        split_dataset(
            args.source_dir, args.output_dir,
            args.train_ratio, args.val_ratio, args.test_ratio,
            args.seed
        )
    
    elif args.command == "augment":
        batch_augment(args.input_dir, args.output_dir, args.num_augment)
    
    elif args.command == "stats":
        visualize_dataset_stats(args.data_dir, args.save_path)
    
    elif args.command == "yaml":
        create_dataset_yaml(args.data_dir, args.output)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

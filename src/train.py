"""
YOLOv8 车辆分类模型训练脚本
============================
功能：
1. 加载并训练 YOLOv8 模型（支持 n/s/m/l/x 各种规模）
2. 支持从预训练权重迁移学习
3. 训练过程自动保存最佳模型和最新模型
4. 支持断点续训
5. 自动记录训练日志
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import torch
from ultralytics import YOLO

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_device():
    """自动检测最佳可用设备"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        return "0"  # Use first GPU
    else:
        print("⚠️  未检测到GPU，将使用CPU训练（速度会很慢）")
        return "cpu"


def get_recommended_config(gpu_mem_gb: float = None) -> dict:
    """
    根据GPU显存大小推荐训练配置
    
    RTX 4050 (6GB) 推荐: YOLOv8s, batch=8, imgsz=640
    RTX 3060 (12GB) 推荐: YOLOv8m, batch=16, imgsz=640
    RTX 3090 (24GB) 推荐: YOLOv8l, batch=32, imgsz=640
    """
    if gpu_mem_gb is None and torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if gpu_mem_gb is None or gpu_mem_gb < 4:
        return {"model": "yolov8n", "batch": 4, "imgsz": 640}
    elif gpu_mem_gb < 8:
        return {"model": "yolov8s", "batch": 8, "imgsz": 640}
    elif gpu_mem_gb < 16:
        return {"model": "yolov8m", "batch": 16, "imgsz": 640}
    elif gpu_mem_gb < 32:
        return {"model": "yolov8l", "batch": 16, "imgsz": 640}
    else:
        return {"model": "yolov8x", "batch": 32, "imgsz": 640}


def load_train_config(config_path: str) -> dict:
    """加载训练配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train(
    data_yaml: str,
    model_name: str = "yolov8s.pt",
    epochs: int = 100,
    batch_size: int = 8,
    img_size: int = 640,
    project: str = None,
    name: str = None,
    resume: bool = False,
    pretrained: bool = True,
    optimizer: str = "AdamW",
    lr0: float = 0.001,
    lrf: float = 0.01,
    patience: int = 20,
    workers: int = 4,
    config_path: str = None,
    **kwargs
):
    """
    训练 YOLOv8 车辆分类模型
    
    Args:
        data_yaml: 数据集配置文件路径 (dataset.yaml)
        model_name: 模型名称或权重路径 (yolov8n/s/m/l/x.pt)
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 输入图像尺寸
        project: 项目保存目录
        name: 实验名称
        resume: 是否断点续训
        pretrained: 是否使用预训练权重
        optimizer: 优化器 (SGD/Adam/AdamW)
        lr0: 初始学习率
        lrf: 最终学习率（lr0 * lrf）
        patience: 早停耐心值
        workers: 数据加载线程数
        config_path: 配置文件路径（优先级高于命令行参数）
    """
    # Load config file if provided
    if config_path and Path(config_path).exists():
        print(f"📄 加载配置文件: {config_path}")
        config = load_train_config(config_path)
        # Config file values as defaults, CLI args override
        model_name = config.get("model", model_name)
        epochs = config.get("epochs", epochs)
        batch_size = config.get("batch_size", batch_size)
        img_size = config.get("img_size", img_size)
        optimizer = config.get("optimizer", optimizer)
        lr0 = config.get("lr0", lr0)
        lrf = config.get("lrf", lrf)
        patience = config.get("patience", patience)
        workers = config.get("workers", workers)
    
    # Default project and name
    if project is None:
        project = str(PROJECT_ROOT / "results")
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"vehicle_cls_{Path(model_name).stem}_{timestamp}"
    
    device = get_device()
    
    print("=" * 60)
    print("  超限车辆智能感知系统 - 模型训练")
    print("=" * 60)
    print(f"📋 训练配置:")
    print(f"  模型: {model_name}")
    print(f"  数据集: {data_yaml}")
    print(f"  轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  图像尺寸: {img_size}")
    print(f"  优化器: {optimizer}")
    print(f"  初始学习率: {lr0}")
    print(f"  设备: {device}")
    print(f"  保存路径: {project}/{name}")
    print("=" * 60)
    
    # Load model
    if resume and Path(project, name, "weights", "last.pt").exists():
        print("📂 从断点恢复训练...")
        model = YOLO(str(Path(project, name, "weights", "last.pt")))
    else:
        print(f"📦 加载模型: {model_name}")
        model = YOLO(model_name)
    
    # Start training
    print("\n🚀 开始训练...\n")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project,
        name=name,
        device=device,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        patience=patience,
        workers=workers,
        pretrained=pretrained,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,      # Generate training plots
        verbose=True,
        exist_ok=True,
        # Data augmentation
        hsv_h=0.015,     # Hue augmentation
        hsv_s=0.7,       # Saturation augmentation
        hsv_v=0.4,       # Value augmentation
        degrees=5.0,     # Rotation augmentation
        translate=0.1,   # Translation augmentation
        scale=0.5,       # Scale augmentation
        fliplr=0.5,      # Horizontal flip
        mosaic=1.0,      # Mosaic augmentation
        mixup=0.1,       # MixUp augmentation
    )
    
    print("\n" + "=" * 60)
    print("  ✅ 训练完成！")
    print("=" * 60)
    print(f"  最佳模型: {project}/{name}/weights/best.pt")
    print(f"  最新模型: {project}/{name}/weights/last.pt")
    print(f"  训练日志: {project}/{name}/")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="超限车辆智能感知系统 - YOLOv8 训练脚本"
    )
    parser.add_argument(
        "--data", required=True,
        help="数据集配置文件路径 (dataset.yaml)"
    )
    parser.add_argument(
        "--model", default="yolov8s.pt",
        help="模型名称或权重路径 (默认: yolov8s.pt)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="训练轮数 (默认: 100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="批次大小 (默认: 根据GPU自动选择)"
    )
    parser.add_argument(
        "--img-size", type=int, default=640,
        help="输入图像尺寸 (默认: 640)"
    )
    parser.add_argument(
        "--project", default=None,
        help="项目保存目录"
    )
    parser.add_argument(
        "--name", default=None,
        help="实验名称"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从断点恢复训练"
    )
    parser.add_argument(
        "--optimizer", default="AdamW",
        choices=["SGD", "Adam", "AdamW"],
        help="优化器 (默认: AdamW)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="初始学习率 (默认: 0.001)"
    )
    parser.add_argument(
        "--patience", type=int, default=20,
        help="早停耐心值 (默认: 20)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="数据加载线程数 (默认: 4)"
    )
    parser.add_argument(
        "--config", default=None,
        help="训练配置文件 (YAML格式)"
    )
    parser.add_argument(
        "--recommend", action="store_true",
        help="显示推荐的训练配置"
    )
    
    args = parser.parse_args()
    
    if args.recommend:
        config = get_recommended_config()
        print("📋 推荐训练配置（基于当前GPU）:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        return
    
    # Auto-detect batch size if not specified
    batch_size = args.batch_size
    if batch_size is None:
        recommended = get_recommended_config()
        batch_size = recommended["batch"]
        print(f"📌 自动选择 batch_size={batch_size}")
    
    train(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=batch_size,
        img_size=args.img_size,
        project=args.project,
        name=args.name,
        resume=args.resume,
        optimizer=args.optimizer,
        lr0=args.lr,
        patience=args.patience,
        workers=args.workers,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()

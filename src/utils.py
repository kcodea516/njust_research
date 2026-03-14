"""
工具函数模块
=============
通用工具函数集合
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).parent.parent


# ============================================================
# 日志配置
# ============================================================
def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    配置日志记录器
    
    Args:
        name: 日志名称
        log_file: 日志文件路径
        level: 日志级别
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


# ============================================================
# GPU 工具
# ============================================================
def get_gpu_info() -> dict:
    """获取GPU信息"""
    if not torch.cuda.is_available():
        return {"available": False, "message": "No GPU detected"}
    
    info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "devices": []
    }
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        device_info = {
            "index": i,
            "name": props.name,
            "total_memory_gb": props.total_memory / 1024**3,
            "compute_capability": f"{props.major}.{props.minor}",
        }
        info["devices"].append(device_info)
    
    return info


def print_gpu_info():
    """打印GPU信息"""
    info = get_gpu_info()
    if not info["available"]:
        print("⚠️  未检测到GPU")
        return
    
    print(f"🖥️  检测到 {info['device_count']} 个GPU:")
    for dev in info["devices"]:
        print(f"  [{dev['index']}] {dev['name']} "
              f"({dev['total_memory_gb']:.1f} GB, "
              f"CC {dev['compute_capability']})")


# ============================================================
# 图像工具
# ============================================================
def resize_image(image: np.ndarray, target_size: int = 640) -> np.ndarray:
    """
    等比缩放图像到目标尺寸
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target_size x target_size
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    
    return canvas


def draw_detections(
    image: np.ndarray,
    detections: list,
    class_names: dict = None,
    line_thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制检测结果
    
    Args:
        image: 输入图像
        detections: 检测结果列表 [{"bbox": [x1,y1,x2,y2], "class_id": int, "confidence": float}]
        class_names: 类别名称字典
        line_thickness: 线条粗细
    """
    # Color palette
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 255), (255, 128, 0), (0, 128, 255),
        (128, 255, 0), (255, 0, 128), (0, 255, 128),
    ]
    
    result = image.copy()
    
    for det in detections:
        bbox = det["bbox"]
        cls_id = det.get("class_id", 0)
        conf = det.get("confidence", 0)
        cls_name = det.get("class_name", "")
        
        if not cls_name and class_names:
            cls_name = class_names.get(cls_id, f"class_{cls_id}")
        
        color = colors[cls_id % len(colors)]
        
        # Draw bounding box
        x1, y1, x2, y2 = [int(c) for c in bbox]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, line_thickness)
        
        # Draw label
        label = f"{cls_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 4), font, font_scale,
                   (255, 255, 255), thickness, cv2.LINE_AA)
    
    return result


# ============================================================
# 实验记录
# ============================================================
class ExperimentLogger:
    """实验记录器 - 记录训练实验的参数和结果"""
    
    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = str(PROJECT_ROOT / "results" / "experiments")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = []
        self._load_existing()
    
    def _load_existing(self):
        """加载已有的实验记录"""
        log_file = self.log_dir / "experiment_log.json"
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                self.experiments = json.load(f)
    
    def log_experiment(
        self,
        name: str,
        model: str,
        dataset: str,
        hyperparams: dict,
        metrics: dict,
        notes: str = ""
    ):
        """记录一次实验"""
        entry = {
            "id": len(self.experiments) + 1,
            "name": name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "dataset": dataset,
            "hyperparams": hyperparams,
            "metrics": metrics,
            "notes": notes,
        }
        self.experiments.append(entry)
        self._save()
        print(f"📝 实验 #{entry['id']} 已记录: {name}")
        return entry
    
    def _save(self):
        """保存实验记录"""
        log_file = self.log_dir / "experiment_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiments, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """打印实验摘要"""
        if not self.experiments:
            print("📭 暂无实验记录")
            return
        
        print(f"\n{'=' * 80}")
        print(f"  📊 实验记录 (共 {len(self.experiments)} 条)")
        print(f"{'=' * 80}")
        
        header = f"{'#':<4} {'名称':<20} {'模型':<15} {'mAP@0.5':<10} {'时间':<20}"
        print(header)
        print("-" * 70)
        
        for exp in self.experiments:
            map50 = exp.get("metrics", {}).get("mAP50", "N/A")
            if isinstance(map50, float):
                map50 = f"{map50:.4f}"
            print(f"{exp['id']:<4} {exp['name']:<20} {exp['model']:<15} "
                  f"{str(map50):<10} {exp['timestamp']:<20}")


# ============================================================
# 数据集工具
# ============================================================
def count_parameters(model) -> dict:
    """计算模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_mb": total * 4 / 1024**2,  # Assuming float32
    }


def set_seed(seed: int = 42):
    """设置随机种子以确保实验可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🎲 随机种子已设置: {seed}")

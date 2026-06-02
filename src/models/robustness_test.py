"""
=========================================
 鲁棒性退化测试脚本
=========================================
核心思路：
1. 拷贝 val 集到临时目录
2. 对图片施加不同退化（夜间/雨/雾/运动模糊）
3. 用 best.pt 在退化后的图片上跑 model.val()
4. 收集 mAP / Precision / Recall / per-class AP
5. 输出对比表格

用法: python robustness_test.py
"""

import os
import sys
import cv2
import shutil
import numpy as np
import random
from pathlib import Path
from ultralytics import YOLO

# ========== 退化函数 ==========

def degrade_night(img, brightness=0.3, noise_std=15):
    """模拟夜间: 降亮度 + 高斯噪声"""
    dark = (img.astype(np.float32) * brightness)
    noise = np.random.normal(0, noise_std, img.shape)
    return np.clip(dark + noise, 0, 255).astype(np.uint8)

def degrade_heavy_night(img):
    """模拟极端夜间: 更暗 + 更大噪声"""
    return degrade_night(img, brightness=0.15, noise_std=25)

def degrade_fog(img):
    """模拟雾天: 高斯模糊 + 白色图层混合"""
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    fog_layer = np.full_like(img, 200, dtype=np.uint8)
    return cv2.addWeighted(blurred, 0.7, fog_layer, 0.3, 0)

def degrade_rain(img):
    """模拟雨天: 斜线 + 降亮度"""
    result = img.copy()
    h, w = result.shape[:2]
    for _ in range(200):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        length = random.randint(15, 30)
        slant = random.randint(-5, 5)
        cv2.line(result, (x, y), (x + slant, y + length), (200, 200, 200), 1)
    return cv2.addWeighted(result, 0.8, np.zeros_like(result), 0.2, 0)

def degrade_motion_blur(img, kernel_size=15):
    """模拟运动模糊: 方向性卷积核"""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size  # 水平方向模糊
    return cv2.filter2D(img, -1, kernel)

def no_degrade(img):
    """不做任何退化 (Clean baseline)"""
    return img

# ========== 退化条件配置 ==========

DEGRADATIONS = {
    "clean":        no_degrade,
    "night":        degrade_night,
    "fog":          degrade_fog,
    "rain":         degrade_rain,
    "motion_blur":  degrade_motion_blur,
    "heavy_night":  degrade_heavy_night,
}

# ========== 主流程 ==========

def create_degraded_dataset(val_img_dir, val_lbl_dir, output_base, degrade_name, degrade_fn):
    """把 val 集图片退化后拷贝到新目录，标签直接复制"""
    out_img = output_base / "images" / "val"
    out_lbl = output_base / "labels" / "val"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # 清理旧文件
    for f in out_img.glob("*"):
        f.unlink()
    for f in out_lbl.glob("*"):
        f.unlink()

    count = 0
    for img_path in sorted(val_img_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 施加退化
        degraded = degrade_fn(img)
        cv2.imwrite(str(out_img / img_path.name), degraded)

        # 标签直接复制（退化不改变框位置）
        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            shutil.copy(lbl_path, out_lbl / lbl_path.name)

        count += 1

    print(f"  ✅ {degrade_name}: 退化了 {count} 张图片")
    return count


def create_temp_yaml(output_base, yaml_path):
    """创建临时的 data.yaml 指向退化后的数据"""
    import yaml
    config = {
        'path': str(output_base),
        'train': 'images/val',  # 不重要，val 测试不需要 train
        'val': 'images/val',
        'names': {0: 'Bus', 1: 'Microbus', 2: 'Minivan', 3: 'Sedan', 4: 'SUV', 5: 'Truck'}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_robustness_test():
    print("=" * 60)
    print("🔬 鲁棒性退化测试 —— 开始")
    print("=" * 60)

    # 路径配置
    model_path = Path("/home/kang/research/src/models/runs/detect/BIT_Vehicle_Model/weights/best.pt")
    val_img_dir = Path("/home/kang/research/data/yolo_dataset/images/val")
    val_lbl_dir = Path("/home/kang/research/data/yolo_dataset/labels/val")
    output_base = Path("/home/kang/research/data/robustness_test")
    results_file = Path("/home/kang/research/src/models/robustness_results.txt")

    if not model_path.exists():
        print(f"❌ 找不到模型权重: {model_path}")
        return
    if not val_img_dir.exists():
        print(f"❌ 找不到验证集: {val_img_dir}")
        return

    model = YOLO(str(model_path))
    class_names = ['Bus', 'Microbus', 'Minivan', 'Sedan', 'SUV', 'Truck']

    all_results = {}

    for deg_name, deg_fn in DEGRADATIONS.items():
        print(f"\n{'='*40}")
        print(f"🌩️ 测试条件: {deg_name}")
        print(f"{'='*40}")

        # 1. 创建退化数据集
        create_degraded_dataset(val_img_dir, val_lbl_dir, output_base, deg_name, deg_fn)

        # 2. 创建临时 yaml
        temp_yaml = output_base / "temp_val.yaml"
        create_temp_yaml(output_base, temp_yaml)

        # 3. 运行验证
        metrics = model.val(
            data=str(temp_yaml),
            imgsz=640,
            batch=16,
            conf=0.001,  # val 时用低阈值，让 PR 曲线更完整
            iou=0.5,
            verbose=False,
        )

        # 4. 提取指标
        result = {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "per_class_ap50": {},
        }

        # Per-class AP
        for i, name in enumerate(class_names):
            if i < len(metrics.box.ap50):
                result["per_class_ap50"][name] = float(metrics.box.ap50[i])

        all_results[deg_name] = result
        print(f"  📊 mAP@0.5={result['mAP50']:.4f}  P={result['precision']:.4f}  R={result['recall']:.4f}")

    # 5. 输出汇总表格
    print("\n" + "=" * 80)
    print("📋 鲁棒性测试结果汇总")
    print("=" * 80)

    header = f"{'条件':<15} {'mAP@0.5':>10} {'mAP@.5:.95':>12} {'Precision':>10} {'Recall':>10} {'Δ mAP@0.5':>12}"
    print(header)
    print("-" * 80)

    clean_map = all_results.get("clean", {}).get("mAP50", 0)

    lines = [header, "-" * 80]
    for deg_name, r in all_results.items():
        delta = r["mAP50"] - clean_map
        line = f"{deg_name:<15} {r['mAP50']:>10.4f} {r['mAP50_95']:>12.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {delta:>+12.4f}"
        print(line)
        lines.append(line)

    # Per-class 表格
    print(f"\n{'='*80}")
    print("📋 Per-class AP@0.5 对比")
    print("=" * 80)
    pc_header = f"{'条件':<15}" + "".join(f"{name:>12}" for name in class_names)
    print(pc_header)
    print("-" * 80)
    lines.extend(["", "Per-class AP@0.5:", pc_header, "-" * 80])

    for deg_name, r in all_results.items():
        vals = "".join(f"{r['per_class_ap50'].get(name, 0):>12.4f}" for name in class_names)
        line = f"{deg_name:<15}{vals}"
        print(line)
        lines.append(line)

    # 保存结果
    with open(results_file, 'w') as f:
        f.write("\n".join(lines))
    print(f"\n✅ 结果已保存到: {results_file}")


if __name__ == "__main__":
    run_robustness_test()

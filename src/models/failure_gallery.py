"""
=========================================
 方向C: 失败案例 Gallery — 面试杀手锏
=========================================
核心思路:
1. 对 val 集中的图片分别在 clean 和 heavy_night 下跑推理
2. 找出 clean 下能检测到、但 heavy_night 下漏检的图片
3. 生成 Gallery 对比图: 左边原图(带检测框) vs 右边退化图(漏检)
4. 按类别统计失败情况

用法: python failure_gallery.py
输出: /home/kang/research/src/models/figures/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import random
from pathlib import Path
from ultralytics import YOLO

# ========== 退化函数 (与 robustness_test.py 一致) ==========

def degrade_heavy_night(img):
    """模拟极端夜间: 亮度降到15% + 高斯噪声σ=25"""
    dark = (img.astype(np.float32) * 0.15)
    noise = np.random.normal(0, 25, img.shape)
    return np.clip(dark + noise, 0, 255).astype(np.uint8)

# ========== 配置 ==========

MODEL_PATH = Path("/home/kang/research/src/models/runs/detect/BIT_Vehicle_Model/weights/best.pt")
VAL_IMG_DIR = Path("/home/kang/research/data/yolo_dataset/images/val")
VAL_LBL_DIR = Path("/home/kang/research/data/yolo_dataset/labels/val")
FIG_DIR = Path("/home/kang/research/src/models/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Bus', 'Microbus', 'Minivan', 'Sedan', 'SUV', 'Truck']

# 颜色方案 (每个类别一种颜色, RGB 0-1)
CLASS_COLORS = {
    'Bus':      '#e74c3c',
    'Microbus': '#3498db',
    'Minivan':  '#2ecc71',
    'Sedan':    '#f39c12',
    'SUV':      '#9b59b6',
    'Truck':    '#1abc9c',
}

CONF_THRESHOLD = 0.5


def get_detections(model, img_path, conf=CONF_THRESHOLD):
    """
    对一张图跑推理，返回检测结果列表
    每个元素: {'cls': 'Sedan', 'conf': 0.92, 'box': [x1, y1, x2, y2]}
    """
    results = model.predict(source=str(img_path), conf=conf, save=False, verbose=False)
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        confidence = float(box.conf[0].item())
        coords = box.xyxy[0].tolist()
        detections.append({
            'cls': cls_name,
            'conf': confidence,
            'box': [int(c) for c in coords]
        })
    return detections


def get_detections_from_array(model, img_array, conf=CONF_THRESHOLD):
    """对 numpy 数组跑推理"""
    results = model.predict(source=img_array, conf=conf, save=False, verbose=False)
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        confidence = float(box.conf[0].item())
        coords = box.xyxy[0].tolist()
        detections.append({
            'cls': cls_name,
            'conf': confidence,
            'box': [int(c) for c in coords]
        })
    return detections


def find_missing_detections(clean_dets, degraded_dets, iou_threshold=0.3):
    """
    找出 clean 中检测到但 degraded 中漏检的目标
    通过 IoU 匹配: 如果 clean 中某个框在 degraded 中找不到 IoU > 阈值的对应框，就是漏检
    """
    def iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    missed = []
    for c_det in clean_dets:
        matched = False
        for d_det in degraded_dets:
            if iou(c_det['box'], d_det['box']) > iou_threshold:
                matched = True
                break
        if not matched:
            missed.append(c_det)
    return missed


def draw_boxes_on_img(img_rgb, detections, missed_boxes=None):
    """
    在图像上绘制检测框
    detections: 正常检测到的框 (实线)
    missed_boxes: 漏检的框 (虚线红框 + ❌)
    """
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(img_rgb)

    # 画检测到的框
    for det in detections:
        x1, y1, x2, y2 = det['box']
        color = CLASS_COLORS.get(det['cls'], '#ffffff')
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        label = f"{det['cls']} {det['conf']:.0%}"
        ax.text(x1, y1 - 5, label, color='white', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.85))

    # 画漏检的框 (红色虚线 + 大叉)
    if missed_boxes:
        for m in missed_boxes:
            x1, y1, x2, y2 = m['box']
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor='red', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            # 画 ❌
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(cx, cy, '✗', color='red', fontsize=22, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.6))
            ax.text(x1, y1 - 5, f"MISSED: {m['cls']}", color='white', fontsize=8,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.85))

    ax.axis('off')
    return fig, ax


def run_failure_gallery():
    print("=" * 60)
    print("🔍 失败案例 Gallery 生成器 — 启动")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"❌ 找不到模型: {MODEL_PATH}")
        return
    if not VAL_IMG_DIR.exists():
        print(f"❌ 找不到验证集: {VAL_IMG_DIR}")
        return

    model = YOLO(str(MODEL_PATH))

    # 扫描所有验证集图片
    all_imgs = sorted(VAL_IMG_DIR.glob("*.jpg"))
    print(f"📂 验证集共 {len(all_imgs)} 张图片")

    # 找失败案例
    failure_cases = []
    miss_stats = {name: 0 for name in CLASS_NAMES}  # 每类漏检次数
    total_stats = {name: 0 for name in CLASS_NAMES}  # 每类总检测次数 (clean下)

    print("🔬 正在逐张对比 Clean vs Heavy Night 检测结果...")

    for i, img_path in enumerate(all_imgs):
        if i % 100 == 0:
            print(f"  进度: {i}/{len(all_imgs)}")

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Clean 推理
        clean_dets = get_detections(model, img_path)
        for d in clean_dets:
            if d['cls'] in total_stats:
                total_stats[d['cls']] += 1

        # Heavy Night 推理
        degraded = degrade_heavy_night(img)
        degraded_dets = get_detections_from_array(model, degraded)

        # 找漏检
        missed = find_missing_detections(clean_dets, degraded_dets)

        if len(missed) > 0:
            for m in missed:
                if m['cls'] in miss_stats:
                    miss_stats[m['cls']] += 1

            failure_cases.append({
                'img_path': img_path,
                'clean_dets': clean_dets,
                'degraded_dets': degraded_dets,
                'missed': missed,
                'n_missed': len(missed),
            })

    print(f"\n📊 扫描完成！共发现 {len(failure_cases)} 张图片存在漏检")

    if len(failure_cases) == 0:
        print("🎉 没有漏检案例 (模型太强了!)")
        return

    # 按漏检数量降序排列，取最严重的案例
    failure_cases.sort(key=lambda x: x['n_missed'], reverse=True)

    # ========== 图 5: 失败案例 Gallery (取 Top-6 最严重的) ==========
    print("\n📸 正在生成失败案例 Gallery...")

    n_cases = min(6, len(failure_cases))
    fig, axes = plt.subplots(n_cases, 2, figsize=(16, 5 * n_cases))

    if n_cases == 1:
        axes = axes.reshape(1, 2)

    for row, case in enumerate(failure_cases[:n_cases]):
        img = cv2.imread(str(case['img_path']))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        degraded = degrade_heavy_night(img)
        degraded_rgb = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)

        # 左: Clean (绿框, 全部检测到)
        ax_clean = axes[row, 0]
        ax_clean.imshow(img_rgb)
        for det in case['clean_dets']:
            x1, y1, x2, y2 = det['box']
            color = CLASS_COLORS.get(det['cls'], '#ffffff')
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor=color, facecolor='none'
            )
            ax_clean.add_patch(rect)
            label = f"{det['cls']} {det['conf']:.0%}"
            ax_clean.text(x1, y1 - 4, label, color='white', fontsize=8, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.85))
        ax_clean.set_title(f"Clean — {len(case['clean_dets'])} detected",
                          fontsize=12, fontweight='bold', color='#2ecc71')
        ax_clean.axis('off')

        # 右: Heavy Night (检测到的 + 漏检的红虚线)
        ax_deg = axes[row, 1]
        ax_deg.imshow(degraded_rgb)
        for det in case['degraded_dets']:
            x1, y1, x2, y2 = det['box']
            color = CLASS_COLORS.get(det['cls'], '#ffffff')
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor=color, facecolor='none'
            )
            ax_deg.add_patch(rect)
            label = f"{det['cls']} {det['conf']:.0%}"
            ax_deg.text(x1, y1 - 4, label, color='white', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.85))
        # 漏检框 (红虚线 + ✗)
        for m in case['missed']:
            x1, y1, x2, y2 = m['box']
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor='red', facecolor='none', linestyle='--'
            )
            ax_deg.add_patch(rect)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ax_deg.text(cx, cy, '✗', color='red', fontsize=20, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.6))
            ax_deg.text(x1, y1 - 4, f"MISSED: {m['cls']}", color='white', fontsize=8,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#e74c3c', alpha=0.85))

        n_detected = len(case['degraded_dets'])
        n_missed = case['n_missed']
        ax_deg.set_title(f"Heavy Night — {n_detected} detected, {n_missed} MISSED",
                        fontsize=12, fontweight='bold', color='#e74c3c')
        ax_deg.axis('off')

    plt.suptitle("Failure Case Gallery: Clean vs Heavy Night Detection",
                fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig5_failure_gallery.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 图5 已保存: fig5_failure_gallery.png")

    # ========== 图 6: 漏检率统计柱状图 ==========
    print("📊 正在生成漏检率统计图...")

    fig, ax = plt.subplots(figsize=(10, 6))

    classes_with_data = []
    miss_rates = []
    miss_counts = []
    total_counts = []

    for name in CLASS_NAMES:
        total = total_stats[name]
        missed = miss_stats[name]
        if total > 0:
            classes_with_data.append(name)
            miss_rates.append(missed / total * 100)
            miss_counts.append(missed)
            total_counts.append(total)

    x = np.arange(len(classes_with_data))
    colors = ['#e74c3c' if r > 20 else '#f39c12' if r > 10 else '#2ecc71' for r in miss_rates]

    bars = ax.bar(x, miss_rates, color=colors, edgecolor='white', linewidth=1.5, width=0.6)

    for bar, rate, mc, tc in zip(bars, miss_rates, miss_counts, total_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{rate:.1f}%\n({mc}/{tc})', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_ylabel('Miss Rate in Heavy Night (%)', fontsize=12)
    ax.set_title('Per-Class Miss Rate Under Heavy Night Degradation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes_with_data, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 添加图例说明颜色
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='High Risk (>20%)'),
        Patch(facecolor='#f39c12', label='Medium Risk (10-20%)'),
        Patch(facecolor='#2ecc71', label='Low Risk (<10%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig6_miss_rate_by_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 图6 已保存: fig6_miss_rate_by_class.png")

    # ========== 打印统计 ==========
    print(f"\n{'=' * 60}")
    print("📋 漏检统计汇总")
    print(f"{'=' * 60}")
    print(f"{'类别':<15} {'Clean总检测':>10} {'HN漏检数':>10} {'漏检率':>10}")
    print("-" * 50)
    for name in CLASS_NAMES:
        total = total_stats[name]
        missed = miss_stats[name]
        rate = (missed / total * 100) if total > 0 else 0
        print(f"{name:<15} {total:>10} {missed:>10} {rate:>9.1f}%")

    print(f"\n🎉 失败案例 Gallery 生成完毕！")
    print(f"👉 图表保存在: {FIG_DIR}")


if __name__ == "__main__":
    run_failure_gallery()

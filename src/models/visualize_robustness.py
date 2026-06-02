"""
=========================================
 鲁棒性实验可视化报告生成脚本
=========================================
生成 3 张专业图表:
1. 各条件下 mAP@0.5 对比柱状图
2. Per-class AP 在 clean vs heavy_night 下的分组对比
3. 退化样例对比（同一张图 6 种退化效果）

用法: python visualize_robustness.py
输出: /home/kang/research/src/models/figures/
"""

import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import cv2
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))
from augmentation import add_noise_and_darken, add_fog_blur, add_rain_lines

# ========== 你的真实实验数据 ==========

conditions = ['Clean', 'Night', 'Fog', 'Rain', 'Motion\nBlur', 'Heavy\nNight']
mAP50 = [0.9862, 0.9718, 0.9866, 0.9855, 0.9849, 0.8002]
precision = [0.9690, 0.9474, 0.9718, 0.9603, 0.9647, 0.8674]
recall = [0.9569, 0.9450, 0.9587, 0.9670, 0.9558, 0.7124]

classes = ['Bus', 'Microbus', 'Minivan', 'Sedan', 'SUV', 'Truck']
ap_clean = [0.9950, 0.9821, 0.9773, 0.9946, 0.9753, 0.9926]
ap_heavy = [0.8748, 0.7246, 0.7097, 0.8606, 0.7822, 0.8492]

# ========== 图表配置 ==========

fig_dir = Path("/home/kang/research/src/models/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

# 专业配色
COLORS = {
    'clean': '#2ecc71',
    'mild': '#3498db',
    'severe': '#e74c3c',
    'highlight': '#f39c12',
}

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
})

# ========== 图 1: mAP@0.5 对比柱状图 ==========

def plot_map_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS['clean'], COLORS['mild'], COLORS['mild'],
              COLORS['mild'], COLORS['mild'], COLORS['severe']]

    bars = ax.bar(conditions, [v * 100 for v in mAP50], color=colors,
                  edgecolor='white', linewidth=1.5, width=0.6)

    # 标注数值
    for bar, val in zip(bars, mAP50):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 标注下降幅度
    clean_val = mAP50[0] * 100
    for i, (bar, val) in enumerate(zip(bars, mAP50)):
        if i > 0:
            delta = (val - mAP50[0]) * 100
            if abs(delta) > 0.1:
                color = '#e74c3c' if delta < 0 else '#27ae60'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 3,
                        f'{delta:+.1f}%', ha='center', va='top', color=color,
                        fontsize=9, fontweight='bold')

    ax.set_ylabel('mAP@0.5 (%)', fontsize=12)
    ax.set_title('YOLOv8 Robustness: mAP@0.5 Under Different Degradations', fontsize=14)
    ax.set_ylim(70, 102)
    ax.axhline(y=clean_val, color='gray', linestyle='--', alpha=0.5, label='Clean baseline')
    ax.legend(loc='lower left')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig1_map_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 图1 已保存: fig1_map_comparison.png")


# ========== 图 2: Per-class AP 对比 ==========

def plot_perclass_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(classes))
    width = 0.35

    bars1 = ax.bar(x - width/2, [v*100 for v in ap_clean], width,
                   label='Clean', color=COLORS['clean'], edgecolor='white')
    bars2 = ax.bar(x + width/2, [v*100 for v in ap_heavy], width,
                   label='Heavy Night', color=COLORS['severe'], edgecolor='white')

    # 标注下降幅度
    for i in range(len(classes)):
        delta = (ap_heavy[i] - ap_clean[i]) * 100
        ax.text(x[i] + width/2, ap_heavy[i]*100 + 1.0,
                f'{delta:.1f}%', ha='center', va='bottom',
                color='#e74c3c', fontsize=9, fontweight='bold')

    ax.set_ylabel('AP@0.5 (%)', fontsize=12)
    ax.set_title('Per-Class AP@0.5: Clean vs Heavy Night', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(60, 105)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig2_perclass_clean_vs_heavy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 图2 已保存: fig2_perclass_clean_vs_heavy.png")


# ========== 图 3: Precision vs Recall 对比 ==========

def plot_precision_recall():
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(conditions))
    width = 0.3

    ax.bar(x - width/2, [v*100 for v in precision], width,
           label='Precision', color='#3498db', edgecolor='white')
    ax.bar(x + width/2, [v*100 for v in recall], width,
           label='Recall', color='#e67e22', edgecolor='white')

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Precision vs Recall Under Different Degradations', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylim(65, 102)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(fig_dir / 'fig3_precision_recall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 图3 已保存: fig3_precision_recall.png")


# ========== 图 4: 退化样例对比 ==========

def plot_degradation_samples():
    """从 val 集取一张图，展示 6 种退化效果"""
    val_dir = Path("/home/kang/research/data/yolo_dataset/images/val")
    sample_imgs = sorted(val_dir.glob("*.jpg"))

    if len(sample_imgs) == 0:
        print("⚠️ Val 目录无图片，跳过样例对比图")
        return

    # 取第一张
    img = cv2.imread(str(sample_imgs[0]))
    if img is None:
        print("⚠️ 无法读取图片")
        return

    # 运动模糊函数
    def motion_blur(img, k=15):
        kernel = np.zeros((k, k))
        kernel[k//2, :] = 1.0 / k
        return cv2.filter2D(img, -1, kernel)

    def heavy_night(img):
        dark = (img.astype(np.float32) * 0.15)
        noise = np.random.normal(0, 25, img.shape)
        return np.clip(dark + noise, 0, 255).astype(np.uint8)

    degradations = {
        'Clean': img,
        'Night': add_noise_and_darken(img),
        'Fog': add_fog_blur(img),
        'Rain': add_rain_lines(img),
        'Motion Blur': motion_blur(img),
        'Heavy Night': heavy_night(img),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (name, deg_img) in zip(axes.flat, degradations.items()):
        # OpenCV BGR → RGB
        ax.imshow(cv2.cvtColor(deg_img, cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Degradation Samples for Robustness Testing', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig4_degradation_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 图4 已保存: fig4_degradation_samples.png (样本: {sample_imgs[0].name})")


# ========== 主入口 ==========

if __name__ == "__main__":
    print("📊 正在生成鲁棒性分析可视化报告...")
    print("=" * 50)
    plot_map_comparison()
    plot_perclass_comparison()
    plot_precision_recall()
    plot_degradation_samples()
    print("=" * 50)
    print(f"🎉 所有图表已保存到: {fig_dir}")
    print("可以直接用在项目报告或 PPT 中！")

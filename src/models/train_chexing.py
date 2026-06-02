import os
import sys
import argparse
import torch

# 必须在导入 ultralytics 之前设置环境变量以强行关闭在线更新检查
os.environ["YOLO_OFFLINE"] = "True"
os.environ["YOLO_SETTINGS_CHECK"] = "False"

from ultralytics import YOLO, settings

# 强行关闭任何 ultralytics 的数据同步与检查功能，确保在无网络隔离环境瞬间启动不卡顿
try:
    settings.update({'sync': False, 'check': False})
except Exception:
    pass

"""
==================================================
 车型细分类模型训练脚本 - train_chexing.py
==================================================
该脚本使用 YOLOv8n 在新生成的 yolo_chexing_dataset（五大类重卡）数据集上进行模型训练。
已全面针对无网络/隔离开发机进行深度优化，关闭任何网络请求，实现瞬间无响应卡顿极速启动！
"""

def train_chexing_model():
    parser = argparse.ArgumentParser(description="YOLOv8 Heavy Vehicle Classification Model Training Script")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    args = parser.parse_args()

    print("🚀 正在检测系统硬件加速设备 (PyTorch CUDA)...")
    cuda_available = torch.cuda.is_available()
    device = 0 if cuda_available else 'cpu'
    
    if cuda_available:
        print(f"🔥 检测到可用 NVIDIA GPU 设备: {torch.cuda.get_device_name(0)}")
        print("⚡ 将使用 GPU (device=0) 启动超高速并行炼丹！")
    else:
        print("⚠️ 未检测到 CUDA 可用设备，或者驱动程序不匹配。")
        print("💻 将使用 CPU 启动模型训练（若想极速收敛，建议在有 CUDA 环境的系统下运行）。")

    print("\n🚀 正在加载 YOLOv8 Nano 预训练模型...")
    pretrained_path = 'yolov8n.pt'
    bit_weights = Path('/home/kang/research/src/models/runs/detect/BIT_Vehicle_Model/weights/best.pt')
    if bit_weights.exists():
        print(f"📂 发现已训练好的 BITVehicle 模型权重 {bit_weights}，将以此为基础进行迁移学习微调！")
        pretrained_path = str(bit_weights)
        
    model = YOLO(pretrained_path)

    print("\n📜 正在读取车型细分类数据集配置文件 (chexing.yaml)...")
    yaml_config_path = "/home/kang/research/data/yolo_chexing_dataset/chexing.yaml"
    
    if not os.path.exists(yaml_config_path):
        print(f"❌ 找不到数据集配置文件: {yaml_config_path}，请确认是否已成功运行 prepare_chexing_dataset.py！")
        sys.exit(1)

    print(f"\n🔥 启动 YOLOv8 训练：epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, device={device}")
    
    # 开始训练
    results = model.train(
        data=yaml_config_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name='Heavy_Vehicle_Model',
        device=device,
        verbose=True
    )
    
    print("\n🎉 训练任务已成功执行完成！")
    print("👉 训练生成的所有模型权重与指标图表保存在此目录下：")
    print("   runs/detect/Heavy_Vehicle_Model/weights/best.pt")

if __name__ == '__main__':
    from pathlib import Path
    train_chexing_model()

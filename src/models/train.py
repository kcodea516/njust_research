import os
from ultralytics import YOLO

"""
=========================================
 阶段三核心实战：开始真实“炼丹” (Model Training)
=========================================
结合之前清洗好的数据 (yaml圣旨) 和 YOLO 预训练大脑 (.pt文件)，
我们要正式训练出属于你自己的第一个极速车辆检测识别大模型！
"""

def train_custom_vehicle_model():
    print("🚀 正在加载 YOLOv8 Nano 版预训练大脑 (Transfer Learning)...")
    # 加载 Ultralytics 官方已经训练好的 yolov8n 大脑底包
    # 它本身认识 80 种物体，拥有极佳的轮廓与边缘感知泛化能力。
    model = YOLO('yolov8n.pt') 

    print("📜 正在读取 BIT-Vehicle 数据集配置文件...")
    # 指向我们在数据处理最后阶段切分好的圣旨文件
    yaml_config_path = "/home/kang/research/data/yolo_dataset/bit_vehicle.yaml"
    
    if not os.path.exists(yaml_config_path):
        print(f"❌ 找不到圣旨文件 {yaml_config_path}，难道你还没运行 split_dataset.py 吗？")
        return

    print("🔥 点火炼丹！模型将开始长达几个小时的反向传播学习过程...")
    # 调用最重要的训练 API 接口
    results = model.train(
        # 1. 指定你的教材 (数据来源)
        data=yaml_config_path,
        
        # 2. 指定 epochs: 让模型把这 12000 张图从头到尾看 30 遍
        # 面试对答点：如果在看第 20 遍时，模型在验证集(val) 上的错误率已经不再下降甚至反弹，我们就叫它【过拟合】。
        epochs=30,
        
        # 3. 指定 batch: 限制显存，由于你平时可能还要开别的软件，batch=16对绝大多数显卡都很安全
        batch=16,
        
        # 4. 指定 imgsz: 将你几千分辨率的大图强行缩放到 640x640，兼顾了速度与显存。
        imgsz=640,
        
        # 5. 指定项目的名字：所有的产出（best.pt, 各种漂亮曲线图）全会保存在 runs/detect/BIT_Vehicle_Model 下
        name='BIT_Vehicle_Model',
        
        # 使用第 0 张显卡 (GPU 序号)，如果你有多个显卡或者是 CPU，这里会相应调整
        device=0, 
        
        # 优化器配置：关闭冗长的打印日志（可选项），加速终端输出
        verbose=True
    )
    
    print("\n🎉 炼丹完毕！这是属于你这几个小时努力的唯一硬核结晶！")
    print("👉 快去看看项目中新生成的这个子文件夹吧： runs/detect/BIT_Vehicle_Model/weights/best.pt")

if __name__ == '__main__':
    train_custom_vehicle_model()

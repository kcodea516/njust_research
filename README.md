# 基于深度学习的超限车辆智能感知方法与系统

## 🚗 项目简介

本项目是南京理工大学科研训练（SRTP）项目，旨在利用深度学习技术构建超限车辆智能感知系统。

**本仓库为车辆分类模块**（子任务一），基于 YOLOv8 实现对不同类型车辆（小轿车、客车、货车、面包车、SUV、危化品运输车）的自动检测与分类。

## 📁 项目结构

```
research/
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   ├── processed/               # 预处理后数据（YOLO格式）
│   └── dataset.yaml             # 数据集配置文件
├── models/                      # 模型权重
├── notebooks/                   # Jupyter 实验记录
├── src/                         # 源代码
│   ├── __init__.py              # 包初始化
│   ├── data_preprocessing.py    # 数据预处理与增强
│   ├── download_dataset.py      # 数据集下载与准备
│   ├── train.py                 # 模型训练
│   ├── evaluate.py              # 模型评估（mAP、FPS等）
│   ├── predict.py               # 推理与可视化
│   └── utils.py                 # 工具函数
├── results/                     # 实验结果
├── configs/                     # 配置文件
│   └── train_config.yaml        # 训练超参数配置
├── docs/                        # 文档
├── setup_env.sh                 # 环境搭建脚本
├── requirements.txt             # Python 依赖
└── README.md                    # 项目说明
```

## 🔧 环境搭建

### 方法一：使用脚本（推荐）
```bash
bash setup_env.sh
```

### 方法二：手动搭建
```bash
# 创建虚拟环境
conda create -n vehicle_perception python=3.10 -y
conda activate vehicle_perception

# 安装 PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 激活环境
```bash
conda activate vehicle_perception
```

### 2. 准备数据集

```bash
# 方式一：创建演示数据集（测试代码流程）
python src/download_dataset.py --dataset demo --output-dir data/processed/demo

# 方式二：使用 COCO 车辆子集
python src/download_dataset.py --dataset coco-vehicle --output-dir data/processed

# 方式三：使用自定义数据集（BIT-Vehicle等）
python src/download_dataset.py --dataset custom --source-dir /path/to/data --output-dir data/processed
```

### 3. 训练模型

```bash
# 使用默认配置训练
python src/train.py --data data/processed/demo/dataset.yaml --model yolov8s.pt --epochs 50

# 使用配置文件训练
python src/train.py --data data/processed/demo/dataset.yaml --config configs/train_config.yaml

# 查看推荐配置
python src/train.py --data data/processed/demo/dataset.yaml --recommend
```

### 4. 评估模型

```bash
# 评估模型性能
python src/evaluate.py eval --model results/xxx/weights/best.pt --data data/processed/demo/dataset.yaml

# 测试推理速度
python src/evaluate.py speed --model results/xxx/weights/best.pt

# 对比多个模型
python src/evaluate.py compare --models model1.pt model2.pt --data data/processed/demo/dataset.yaml

# 单张图片推理
python src/evaluate.py predict --model results/xxx/weights/best.pt --image test.jpg
```

### 5. 批量推理

```bash
# 对目录中的图片进行推理
python src/predict.py images --model results/xxx/weights/best.pt --input-dir test_images/

# 对视频进行推理
python src/predict.py video --model results/xxx/weights/best.pt --video test_video.mp4
```

### 6. 数据预处理

```bash
# 验证数据集
python src/data_preprocessing.py validate --data-dir data/processed

# 数据集划分
python src/data_preprocessing.py split --source-dir data/raw --output-dir data/processed

# 数据增强
python src/data_preprocessing.py augment --input-dir data/processed/images/train --output-dir data/augmented

# 统计可视化
python src/data_preprocessing.py stats --data-dir data/processed
```

## 🏗️ 技术架构

- **模型框架**: YOLOv8 (Ultralytics)
- **深度学习**: PyTorch
- **预训练**: COCO 数据集迁移学习
- **注意力机制**: Transformer (YOLOv8 内置)
- **数据增强**: Mosaic、MixUp、随机亮度/对比度/模糊等

## 📊 车辆分类类别

| ID | 英文 | 中文 | 说明 |
|----|------|------|------|
| 0  | car    | 小轿车 | 普通家用轿车 |
| 1  | bus    | 客车   | 大型客运车辆 |
| 2  | truck  | 货车   | 运输货物车辆（重点超限对象） |
| 3  | van    | 面包车 | 厢式货车/小型商用车 |
| 4  | suv    | SUV    | 运动型多用途车 |
| 5  | tanker | 危化品运输车 | 罐车/危险品运输（重点监控对象） |

## 👥 团队分工

| 成员 | 负责模块 |
|------|---------|
| 康思源 | **车辆分类模型构建与实验**（本仓库） |
| 王晨宇 | 车牌识别模型和系统构建 |
| 顾倚嘉 | 车辆超载检测 |

## 📄 License

南京理工大学科研训练项目，仅限学术用途。
# njust_research

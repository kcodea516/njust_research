#!/bin/bash
# ============================================================
# 超限车辆智能感知系统 - 环境搭建脚本
# ============================================================

set -e

ENV_NAME="vehicle_perception"
PYTHON_VERSION="3.10"

echo "============================================"
echo "  超限车辆智能感知系统 - 环境搭建"
echo "============================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ conda 未安装，请先安装 Miniconda 或 Anaconda"
    exit 1
fi

# Create conda environment
echo ""
echo "📦 创建 conda 环境: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate environment
echo ""
echo "🔧 激活环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install PyTorch with CUDA
echo ""
echo "🔥 安装 PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo ""
echo "📚 安装其他依赖..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "============================================"
echo "  ✅ 环境搭建完成！验证安装..."
echo "============================================"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

python -c "
from ultralytics import YOLO
print('YOLOv8 (Ultralytics) installed successfully ✅')
"

echo ""
echo "============================================"
echo "  🎉 所有环境搭建完成！"
echo "  使用方式: conda activate $ENV_NAME"
echo "============================================"

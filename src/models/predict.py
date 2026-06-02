import os
import random
from pathlib import Path
from ultralytics import YOLO

"""
=========================================
 阶段四：实战验证 (Baseline Inference)
=========================================
咱们的模型昨天跑完了 30 轮，最好的快照被存在了 best.pt。
今天这个脚本，就是“出师大考”：
我们不给它看训练集，而是从“被藏起来”的 20% 验证集里盲抽一张，
看看我们的模型究竟能不能准确画框并且认出它的所属车型！
"""

def run_prediction():
    # 1. 挂载我们刚刚炼出来的神级大脑
    weights_path = "/home/kang/research/src/models/runs/detect/BIT_Vehicle_Model/weights/best.pt"
    
    if not os.path.exists(weights_path):
        print(f"❌ 找不到权重文件 {weights_path}。是不是你名字填错了，或训练没跑完？")
        return
        
    print("🧠 正在装载我们提炼的 best.pt 视觉大脑...")
    model = YOLO(weights_path)
    
    # 2. 从我们之前切分出来的纯净 val 库里随机抓一张图
    val_images_dir = Path("/home/kang/research/data/yolo_dataset/images/val")
    all_val_imgs = list(val_images_dir.glob("*.jpg"))
    
    if len(all_val_imgs) == 0:
        print("❌ Val 文件夹里没有图！")
        return
        
    random_test_img = random.choice(all_val_imgs)
    print(f"📸 选中的盲测对象: {random_test_img.name}")
    
    # 3. 核心 API：一句话完成前向推理 (Forward Pass)
    # save=True 表示自动帮我们在原图上画框并存下来
    # conf=0.5 表示：模型必须有 50% 以上的把握，才敢把框画出来 (抗干扰机制)
    results = model.predict(source=str(random_test_img), save=True, conf=0.5)
    
    print("\n✅ 推理完成！")
    # Ultralytics 每次跑 predict 都会在 runs/detect/predict[1、2、3...] 下建新文件夹
    save_dir = results[0].save_dir
    print(f"👉 【强烈建议】立刻去文件浏览器里双击打开这个目录下的图片看看实力: {save_dir}")

if __name__ == "__main__":
    run_prediction()

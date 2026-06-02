from ultralytics import YOLO
import os

"""
=========================================
 核心实战：断点续训 (Resume Training)
=========================================
当遇到 "Killed" (内存溢出) 或者停电等意外中断时，
我们不需要重头再来，利用此脚本从上次摔倒的地方爬起来继续炼丹！
"""

def resume_my_training():
    last_weights = "runs/detect/BIT_Vehicle_Model/weights/last.pt"
    
    if not os.path.exists(last_weights):
        print(f"❌ 找不到上次的快照文件: {last_weights}")
        return

    print(f"🚀 正在恢复训练！目标：完成剩余的 Epochs...")
    
    # 1. 加载最后一次保存的检查点
    model = YOLO(last_weights)
    
    # 2. 只有一句话命令：resume=True
    # 我们调低了 batch=8 并设置 workers=0，这能极大地释放内存，确保不再被系统杀掉
    model.train(
        resume=True,
        batch=8,    # 降低每批处理量，防 OOM
        workers=0   # 减少多线程内存开销，稳字当头
    )

if __name__ == '__main__':
    resume_my_training()

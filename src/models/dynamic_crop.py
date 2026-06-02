import os
from pathlib import Path
from ultralytics import YOLO
import cv2  # OpenCV 用于底层图像矩阵裁切

"""
=========================================
 阶段五 5.1实战：Stage-1 动态 ROI 裁剪模块
=========================================
本脚本不仅仅是画个框，更是实打实的“数据分流路由器”。
它会利用模型生成的精确定位，运用 Numpy 矩阵切片，
把每辆车单独“抠”成一张纯净的高清图片，为稍后的车牌识别(Stage-2)提供子弹。
"""

def extract_and_crop():
    # 1. 挂载我们最强的一期大脑 best.pt
    model_path = "/home/kang/research/src/models/runs/detect/BIT_Vehicle_Model/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ 找不到权重文件 {model_path}")
        return
        
    print("🧠 装载神级视觉感知中枢 (Stage-1 Detection)...")
    model = YOLO(model_path)
    
    # 2. 从验证集里找一张复杂的“考卷”
    # 注意：你可以随时替换成你在真实公路上拍的中国大卡车图片！
    val_images_dir = Path("/home/kang/research/data/yolo_dataset/images/val")
    # 抽取一张固定的样本做演示，或者写个随机抽取
    all_imgs = list(val_images_dir.glob("*.jpg"))
    if not all_imgs: return
    test_img_path = all_imgs[0] 
    print(f"📸 开始猎杀目标图片: {test_img_path.name}")
    
    # 用 OpenCV 把原图读取为 Numpy 矩阵矩阵
    # 格式为: original_img_matrix[高度(H), 宽度(W), 通道(C)]
    original_img_matrix = cv2.imread(str(test_img_path))
    
    # 3. 核心 API: 进行前向雷达扫描
    # 不在原图画框了，我要纯净的数据！
    results = model.predict(source=str(test_img_path), conf=0.5, save=False)
    
    # 准备下级网络的“进料口”文件夹
    crop_output_dir = Path("/home/kang/research/src/models/cropped_vehicles")
    crop_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 清理掉之前切的旧图
    for old_file in crop_output_dir.glob("*.jpg"):
        old_file.unlink()
    
    # 4. 暴力提取与切割！
    # results[0].boxes 里存着检测到的所有目标的数学坐标
    boxes = results[0].boxes
    print(f"\n🎯 报告长官：在这张图中扫描到了 {len(boxes)} 辆载具！准备实施分离手术...")
    
    for idx, box in enumerate(boxes):
        # 将张量 (Tensor) 转成普通的 python list [x1, y1, x2, y2]
        # x1,y1 是左上角坐标；x2,y2 是右下角坐标
        coords = box.xyxy[0].tolist() 
        x1, y1, x2, y2 = map(int, coords)
        
        # 提取这是什么车类别 (比如 Sedan, Truck)
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        
        print(f"    ✂️ 正在切除背景... [{cls_name}] (坐标: {x1},{y1} 到 {x2},{y2})")
        
        # 🔥 面试极高含金量的核心代码：Numpy 矩阵切片手术！ 🔥
        # 注意：图像在 CV 里的切片规则是 先 高度(Y轴) 后 宽度(X轴)
        try:
            cropped_matrix = original_img_matrix[y1:y2, x1:x2]
            
            # 保存手术取出的纯净的卡车/轿车图片，命名带上车型
            crop_filename = crop_output_dir / f"vehicle_{idx}_{cls_name}.jpg"
            cv2.imwrite(str(crop_filename), cropped_matrix)
        except Exception as e:
            print(f"切片失败: {e}")
            
    print(f"\n==========================================")
    print(f"✅ Stage-1 级联分流成功！请在左侧目录检查你的战利品：")
    print(f"👉 /home/kang/research/src/models/cropped_vehicles")
    print(f"这些干净无背景的车辆图，将是明天 Stage-2 车牌扫描引擎最完美的饵料！")
    print(f"==========================================")

if __name__ == "__main__":
    extract_and_crop()

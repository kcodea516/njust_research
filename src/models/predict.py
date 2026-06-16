import os
import random
from pathlib import Path
from ultralytics import YOLO

def run_prediction():
    weights_path = "/home/kang/research/src/models/runs/detect/BIT_Vehicle_Model/weights/best.pt"
    
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return
        
    print("Loading model weights...")
    model = YOLO(weights_path)
    
    val_images_dir = Path("/home/kang/research/data/yolo_dataset/images/val")
    all_val_imgs = list(val_images_dir.glob("*.jpg"))
    
    if len(all_val_imgs) == 0:
        print("No validation images found.")
        return
        
    random_test_img = random.choice(all_val_imgs)
    print(f"Selected validation image: {random_test_img.name}")
    
    results = model.predict(source=str(random_test_img), save=True, conf=0.5)
    
    print("Prediction finished.")
    save_dir = results[0].save_dir
    print(f"Annotated image directory: {save_dir}")

if __name__ == "__main__":
    run_prediction()

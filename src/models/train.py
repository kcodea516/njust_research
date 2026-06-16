import os
from ultralytics import YOLO

def train_custom_vehicle_model():
    print("Loading YOLOv8 Nano pretrained weights...")
    model = YOLO("yolov8n.pt")

    print("Reading BIT-Vehicle dataset config...")
    yaml_config_path = "/home/kang/research/data/yolo_dataset/bit_vehicle.yaml"
    
    if not os.path.exists(yaml_config_path):
        print(f"Dataset config not found: {yaml_config_path}")
        return

    print("Starting training...")
    model.train(
        data=yaml_config_path,
        epochs=30,
        batch=16,
        imgsz=640,
        name="BIT_Vehicle_Model",
        device=0,
        verbose=True,
    )
    
    print("Training finished. Best weights: runs/detect/BIT_Vehicle_Model/weights/best.pt")

if __name__ == "__main__":
    train_custom_vehicle_model()

import os
from pathlib import Path
from typing import Any


class VehicleTypeClassifier:
    """Five-class heavy-truck detector wrapper."""

    DEFAULT_WEIGHTS = Path("/home/kang/research/src/models/runs/detect/Heavy_Vehicle_Model/weights/best.pt")
    
    HEAVY_TRUCK_MAP = {
        "Kong": "空载重卡",
        "Sanhuo": "散货重卡",
        "Jizhuangxiang": "集装箱重卡",
        "Caoguan": "槽罐重卡",
        "Zhatu": "渣土重卡"
    }

    def __init__(self, weights_path: Path | None = None) -> None:
        self.weights_path = weights_path or self.DEFAULT_WEIGHTS
        self.model: Any = None
        self.load_error: str | None = None
        self.module_status = "connected"

    def _load_model(self) -> Any:
        if self.model is not None:
            return self.model

        if not self.weights_path.exists():
            self.load_error = f"未找到车型细分类权重文件: {self.weights_path}"
            return None

        try:
            os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights_path))
            self.load_error = None
            return self.model
        except Exception as exc:
            self.load_error = f"车型细分类模型加载失败: {exc}"
            return None

    def predict(self, image_path: Path, primary_vehicle: dict | None = None) -> dict:
        if not primary_vehicle:
            return {
                "vehicle_type": None,
                "vehicle_type_confidence": None,
                "module_status": "no_detection",
                "module_message": "未检测到可用于判定车型的车辆目标",
            }

        model = self._load_model()
        class_name = primary_vehicle.get("class_name")
        class_label = primary_vehicle.get("class_label") or class_name
        
        if model is None or class_name != "Truck":
            module_message = "车型特征提取已就绪"
            if self.load_error:
                module_message = "车型细分类模型暂未接入，当前沿用基础车辆类别"
            
            vehicle_type = "货车" if class_name == "Truck" else class_label
            return {
                "vehicle_type": vehicle_type,
                "vehicle_type_confidence": primary_vehicle.get("confidence"),
                "module_status": "connected" if model else "derived",
                "module_message": module_message,
            }

        try:
            results = model.predict(source=str(image_path), conf=0.25, save=False, verbose=False)
            result = results[0]
            names = result.names
            
            # 寻找置信度最高的目标
            best_cls_id = None
            best_conf = 0.0
            
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                if confidence > best_conf:
                    best_conf = confidence
                    best_cls_id = cls_id
            
            if best_cls_id is not None:
                detected_class = names.get(best_cls_id, str(best_cls_id))
                vehicle_type = self.HEAVY_TRUCK_MAP.get(detected_class, f"{detected_class}型货车")
                return {
                    "vehicle_type": vehicle_type,
                    "vehicle_type_confidence": round(best_conf, 4),
                    "module_status": "connected",
                    "module_message": "车型细分类模型已接入",
                }
            else:
                return {
                    "vehicle_type": "重型货车",
                    "vehicle_type_confidence": primary_vehicle.get("confidence"),
                    "module_status": "connected",
                    "module_message": "车型细分类模型未检出目标，当前沿用基础车辆类别",
                }
                
        except Exception as exc:
            return {
                "vehicle_type": "货车",
                "vehicle_type_confidence": primary_vehicle.get("confidence"),
                "module_status": "error",
                "module_message": f"车型细分类推理异常: {exc}",
            }

import os
from pathlib import Path
from typing import Any


class VehicleDetector:
    """封装当前已训练的 YOLO 目标检测模型。"""

    DEFAULT_WEIGHTS = Path("/home/kang/research/src/models/runs/detect/BIT_Vehicle_Model/weights/best.pt")
    LABEL_MAP = {
        "Bus": "客车",
        "Microbus": "小型客车",
        "Minivan": "面包车",
        "Sedan": "轿车",
        "SUV": "SUV",
        "Truck": "货车",
    }

    def __init__(self, weights_path: Path | None = None, conf_threshold: float = 0.5) -> None:
        self.weights_path = weights_path or self.DEFAULT_WEIGHTS
        self.conf_threshold = conf_threshold
        self.model: Any = None
        self.load_error: str | None = None

    def _load_model(self) -> Any:
        if self.model is not None:
            return self.model

        if not self.weights_path.exists():
            self.load_error = f"未找到目标检测权重文件: {self.weights_path}"
            return None

        try:
            os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))
            from ultralytics import YOLO

            self.model = YOLO(str(self.weights_path))
            self.load_error = None
            return self.model
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"目标检测依赖或模型加载失败: {exc}"
            return None

    def predict(self, image_path: Path, annotated_dir: Path, conf_threshold: float | None = None) -> dict:
        model = self._load_model()
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        if model is None:
            return {
                "module_status": "not_ready",
                "module_message": self.load_error or "目标检测模型未就绪",
                "conf_threshold": threshold,
                "detections": [],
                "primary_vehicle": None,
                "annotated_image_url": None,
            }

        try:
            results = model.predict(source=str(image_path), conf=threshold, save=False, verbose=False)
        except Exception as exc:  # noqa: BLE001
            return {
                "module_status": "error",
                "module_message": f"目标检测推理失败: {exc}",
                "conf_threshold": threshold,
                "detections": [],
                "primary_vehicle": None,
                "annotated_image_url": None,
            }

        result = results[0]
        names = result.names
        detections = []

        for index, box in enumerate(result.boxes):
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [round(float(value), 2) for value in box.xyxy[0].tolist()]
            class_name = names.get(cls_id, str(cls_id))
            detections.append(
                {
                    "id": index + 1,
                    "class_name": class_name,
                    "class_label": self.LABEL_MAP.get(class_name, class_name),
                    "confidence": round(confidence, 4),
                    "bbox": [x1, y1, x2, y2],
                }
            )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        primary_vehicle = detections[0] if detections else None
        annotated_url = self._save_annotated_image(result, image_path, annotated_dir)

        return {
            "module_status": "connected",
            "module_message": self._build_module_message(len(detections), threshold),
            "conf_threshold": threshold,
            "model_scope": "当前车辆检测模型支持类别为客车、小型客车、面包车、轿车、SUV、货车；非高速车辆视角、极端遮挡、过小目标等图片可能检出率下降。",
            "detections": detections,
            "primary_vehicle": primary_vehicle,
            "annotated_image_url": annotated_url,
        }

    def _build_module_message(self, detection_count: int, threshold: float) -> str:
        if detection_count:
            return f"已接入车辆目标检测模型，当前阈值 {threshold:.2f}"
        return f"未检出车辆目标。可尝试降低阈值，或使用更接近高速监控视角的车辆图片。当前阈值 {threshold:.2f}"

    def _save_annotated_image(self, result: Any, image_path: Path, annotated_dir: Path) -> str | None:
        annotated_dir.mkdir(parents=True, exist_ok=True)
        output_path = annotated_dir / f"{image_path.stem}_detected.jpg"

        try:
            import cv2

            plotted = result.plot()
            cv2.imwrite(str(output_path), plotted)
            return f"/outputs/{output_path.name}"
        except Exception:
            return None

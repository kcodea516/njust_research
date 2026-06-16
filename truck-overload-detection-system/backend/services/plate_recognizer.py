from pathlib import Path


class PlateRecognizer:
    """车牌识别服务占位，等待真实检测和 OCR 模型接入。"""

    def __init__(self) -> None:
        self.module_status = "pending"

    def predict(self, image_path: Path) -> dict:
        return {
            "plate_number": None,
            "plate_confidence": None,
            "module_status": self.module_status,
            "module_message": "车牌检测与 OCR 识别模型暂未接入",
        }

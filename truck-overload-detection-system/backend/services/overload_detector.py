from pathlib import Path


class OverloadDetector:
    """超载检测服务占位，等待真实规则、轴数和称重相关模型接入。"""

    def __init__(self) -> None:
        self.module_status = "pending"

    def predict(self, image_path: Path, detections: list[dict] | None = None, vehicle_type_label: str | None = None) -> dict:
        return {
            "overload_suspected": None,
            "overload_confidence": None,
            "risk_level": None,
            "limit_weight": None,
            "actual_weight": None,
            "overload_ratio": None,
            "axles": None,
            "basis": "超载检测模型暂未接入，当前不输出重量或超载结论。",
            "module_status": self.module_status,
            "module_message": "超载检测模型暂未接入",
        }

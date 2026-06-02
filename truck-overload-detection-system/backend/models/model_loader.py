from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ModelConfig:
    """模型配置占位，后续可扩展为 PyTorch、ONNX 或 OpenCV 模型路径。"""

    name: str
    model_path: Optional[Path] = None
    backend: str = "mock"


class ModelLoader:
    """统一管理模型加载逻辑，避免业务服务直接依赖具体框架。"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: Any = None

    def load(self) -> Any:
        if self.config.backend == "mock":
            self.model = {"name": self.config.name, "backend": "mock"}
            return self.model

        # 后续可在这里接入真实模型，例如：
        # if self.config.backend == "pytorch":
        #     import torch
        #     self.model = torch.load(self.config.model_path, map_location="cpu")
        # elif self.config.backend == "onnx":
        #     import onnxruntime as ort
        #     self.model = ort.InferenceSession(str(self.config.model_path))
        raise NotImplementedError(f"暂未实现的模型后端: {self.config.backend}")

    def get_model(self) -> Any:
        if self.model is None:
            return self.load()
        return self.model

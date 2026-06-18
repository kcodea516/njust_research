from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class FocalBCEWithLogitsLoss(nn.Module):
    """BCEWithLogits with focal modulation, returning unreduced element losses."""

    def __init__(self, gamma: float = 1.5, alpha: float | Iterable[float] | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))
        else:
            values = [float(x) for x in alpha]
            self.register_buffer("alpha", torch.tensor(values, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pred_prob = pred.sigmoid()
        p_t = target * pred_prob + (1.0 - target) * (1.0 - pred_prob)
        loss = loss * (1.0 - p_t).clamp(min=0.0, max=1.0).pow(self.gamma)

        alpha = getattr(self, "alpha", None)
        if alpha is not None:
            alpha = alpha.to(device=pred.device, dtype=pred.dtype)
            if alpha.ndim == 1:
                if alpha.numel() != pred.shape[-1]:
                    raise ValueError(f"Focal alpha length {alpha.numel()} does not match nc={pred.shape[-1]}")
                alpha = alpha.view(1, 1, -1)
            loss = loss * (target * alpha + (1.0 - target) * (1.0 - alpha))

        return loss


def parse_focal_alpha(value: str | None, data: Path, power: float = 0.5) -> float | list[float] | None:
    """Parse focal alpha from CLI text.

    Accepted values:
    - none/off/disable: no alpha factor
    - auto: compute per-class positive alpha from train labels
    - scalar float: one alpha for all classes
    - comma list: per-class alpha values, e.g. 0.35,0.25,0.45,0.70,0.75
    """

    if value is None:
        return None

    text = value.strip().lower()
    if text in {"", "none", "off", "false", "disable", "disabled"}:
        return None
    if text == "auto":
        return compute_auto_alpha(data, power=power)
    if "," in text:
        return [float(part.strip()) for part in text.split(",") if part.strip()]
    return float(text)


def compute_auto_alpha(data_yaml: Path, power: float = 0.5) -> list[float]:
    """Compute per-class focal alpha from YOLO train-label counts."""

    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if isinstance(names, dict):
        nc = len(names)
    elif isinstance(names, list):
        nc = len(names)
    else:
        raise ValueError(f"Could not read class names from {data_yaml}")

    train_images = _resolve_split_path(data_yaml, data, "train")
    train_labels = _images_to_labels_path(train_images)
    counts = count_yolo_labels(train_labels, nc)
    if not any(counts):
        raise ValueError(f"No YOLO labels found under {train_labels}")

    weights = []
    max_count = max(c for c in counts if c > 0)
    for count in counts:
        ratio = max_count / max(count, 1)
        weights.append(ratio**power)

    min_w, max_w = min(weights), max(weights)
    if max_w == min_w:
        return [0.25 for _ in weights]

    min_alpha, max_alpha = 0.25, 0.75
    return [min_alpha + (w - min_w) / (max_w - min_w) * (max_alpha - min_alpha) for w in weights]


def count_yolo_labels(labels_dir: Path, nc: int) -> list[int]:
    counts = [0 for _ in range(nc)]
    for label_file in labels_dir.rglob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(float(parts[0]))
            if 0 <= cls_id < nc:
                counts[cls_id] += 1
    return counts


def enable_focal_loss(gamma: float, alpha: float | Iterable[float] | None = None) -> None:
    """Patch Ultralytics detection loss so cls BCE uses focal BCE."""

    from ultralytics.utils import loss as loss_module

    loss_cls = loss_module.v8DetectionLoss
    original_init = getattr(loss_cls, "_njust_original_init", loss_cls.__init__)

    def patched_init(self, model, *args, **kwargs):
        original_init(self, model, *args, **kwargs)
        self.bce = FocalBCEWithLogitsLoss(gamma=gamma, alpha=alpha).to(self.device)

    loss_cls._njust_original_init = original_init
    loss_cls.__init__ = patched_init


def _resolve_split_path(data_yaml: Path, data: dict, split: str) -> Path:
    root = Path(data.get("path", data_yaml.parent))
    split_value = Path(data[split])
    if split_value.is_absolute():
        return split_value
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root / split_value


def _images_to_labels_path(images_path: Path) -> Path:
    parts = list(images_path.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts)
    return images_path.parent / "labels" / images_path.name

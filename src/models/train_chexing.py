import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO, settings


os.environ.setdefault("YOLO_OFFLINE", "True")
os.environ.setdefault("YOLO_SETTINGS_CHECK", "False")

try:
    settings.update({"sync": False, "check": False})
except Exception:
    pass


PROJECT_ROOT = Path("/home/kang/research")
DEFAULT_DATA = PROJECT_ROOT / "data/yolo_chexing_dataset/chexing.yaml"
DEFAULT_BASE_WEIGHTS = PROJECT_ROOT / "src/models/runs/detect/BIT_Vehicle_Model/weights/best.pt"
FALLBACK_WEIGHTS = PROJECT_ROOT / "src/models/yolov8n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a five-class heavy-truck YOLO detector.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to chexing.yaml.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_BASE_WEIGHTS, help="Initial weights.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Ultralytics device value, for example 0, 0,1, or cpu.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "src/models/runs/detect")
    parser.add_argument("--name", default="Heavy_Vehicle_Model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true", help="Cache images if server memory is enough.")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine learning-rate schedule.")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--optimizer", default="auto")
    return parser.parse_args()


def resolve_weights(path: Path) -> Path | str:
    if path.exists():
        return path
    if FALLBACK_WEIGHTS.exists():
        print(f"Initial weights not found: {path}. Falling back to {FALLBACK_WEIGHTS}.")
        return FALLBACK_WEIGHTS
    print(f"Initial weights not found: {path}. Falling back to Ultralytics yolov8n.pt name.")
    return "yolov8n.pt"


def resolve_device(device_arg: str | None) -> str | int:
    if device_arg:
        return int(device_arg) if device_arg.isdigit() else device_arg
    return 0 if torch.cuda.is_available() else "cpu"


def train() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    weights = resolve_weights(args.weights)
    device = resolve_device(args.device)
    print(f"Data: {args.data}")
    print(f"Weights: {weights}")
    print(f"Device: {device}")

    model = YOLO(str(weights))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        cache=args.cache,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        lr0=args.lr0,
        optimizer=args.optimizer,
        verbose=True,
    )


if __name__ == "__main__":
    train()

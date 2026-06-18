import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO, settings

from loss_patches import enable_focal_loss, parse_focal_alpha


os.environ.setdefault("YOLO_OFFLINE", "True")
os.environ.setdefault("YOLO_SETTINGS_CHECK", "False")

try:
    settings.update({"sync": False, "check": False})
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data/yolo_chexing_dataset/chexing.yaml"
DEFAULT_BASE_WEIGHTS = PROJECT_ROOT / "weights/pretrained/yolo11m.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a five-class heavy-truck YOLO detector.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to chexing.yaml.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_BASE_WEIGHTS, help="Initial YOLO weights.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Ultralytics device value, for example 0, 0,1, or cpu.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "src/models/runs/detect")
    parser.add_argument("--name", default="Vehicle5_YOLO11m_2x3090")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true", help="Cache images if server memory is enough.")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine learning-rate schedule.")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--loss", choices=("bce", "focal"), default="bce", help="Classification loss variant.")
    parser.add_argument("--focal-gamma", type=float, default=1.5, help="Focal-loss focusing parameter.")
    parser.add_argument(
        "--focal-alpha",
        default="none",
        help="Focal alpha: none, auto, scalar, or comma-separated per-class values.",
    )
    parser.add_argument(
        "--focal-alpha-power",
        type=float,
        default=0.5,
        help="Power used when --focal-alpha auto converts class counts to alpha values.",
    )
    return parser.parse_args()


def resolve_weights(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(
        f"YOLO11 weights not found: {path}\n"
        "Please download yolo11m.pt to weights/pretrained/yolo11m.pt first."
    )


def resolve_device(device_arg: str | None) -> str | int:
    if device_arg:
        return int(device_arg) if device_arg.isdigit() else device_arg
    return "0,1" if torch.cuda.device_count() >= 2 else (0 if torch.cuda.is_available() else "cpu")


def train() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    weights = resolve_weights(args.weights)
    device = resolve_device(args.device)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data: {args.data}")
    print(f"Weights: {weights}")
    print(f"Device: {device}")
    print(f"Loss: {args.loss}")

    if args.loss == "focal":
        alpha = parse_focal_alpha(args.focal_alpha, args.data, power=args.focal_alpha_power)
        print(f"Focal gamma: {args.focal_gamma}")
        print(f"Focal alpha: {alpha}")
        if isinstance(device, str) and "," in device:
            print("Warning: focal loss is enabled by a runtime patch; verify DDP results before using multi-GPU runs.")
        enable_focal_loss(gamma=args.focal_gamma, alpha=alpha)

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

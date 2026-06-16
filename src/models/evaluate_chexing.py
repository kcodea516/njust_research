import argparse
import os
from pathlib import Path

from ultralytics import YOLO, settings


os.environ.setdefault("YOLO_OFFLINE", "True")
os.environ.setdefault("YOLO_SETTINGS_CHECK", "False")

try:
    settings.update({"sync": False, "check": False})
except Exception:
    pass


PROJECT_ROOT = Path("/home/kang/research")
DEFAULT_DATA = PROJECT_ROOT / "data/yolo_chexing_dataset/chexing.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the five-class heavy-truck YOLO detector.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained best.pt.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to chexing.yaml.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "src/models/runs/detect")
    parser.add_argument("--name", default="Heavy_Vehicle_Model_val")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=True,
        verbose=True,
    )

    box = metrics.box
    print("Validation summary")
    print(f"mAP50: {box.map50:.4f}")
    print(f"mAP50-95: {box.map:.4f}")
    print(f"mean precision: {box.mp:.4f}")
    print(f"mean recall: {box.mr:.4f}")

    names = model.names
    for cls_id, ap50 in enumerate(box.ap50):
        name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        ap = box.ap[cls_id]
        print(f"{name}: AP50={ap50:.4f}, AP50-95={ap:.4f}")


if __name__ == "__main__":
    main()

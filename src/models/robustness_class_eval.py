import argparse
import csv
import os
import shutil
from pathlib import Path

import yaml
from PIL import Image, ImageEnhance, ImageFilter
from ultralytics import YOLO, settings


os.environ.setdefault("YOLO_OFFLINE", "True")
os.environ.setdefault("YOLO_SETTINGS_CHECK", "False")

try:
    settings.update({"sync": False, "check": False})
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data/yolo_chexing_dataset/chexing.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "src/models/runs/robustness"


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate per-class robustness under image degradations.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained best.pt.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to dataset yaml.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="Vehicle5_class_robustness")
    parser.add_argument(
        "--variants",
        default="clean,low_light,overexposure,gaussian_blur,jpeg",
        help="Comma-separated variants: clean, low_light, overexposure, gaussian_blur, jpeg.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Optional debug limit. 0 means all val images.")
    parser.add_argument("--low-light-factor", type=float, default=0.45)
    parser.add_argument("--overexposure-factor", type=float, default=1.70)
    parser.add_argument("--blur-radius", type=float, default=2.0)
    parser.add_argument("--jpeg-quality", type=int, default=35)
    return parser.parse_args()


def load_data_yaml(data_yaml: Path) -> dict:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml: {data_yaml}")
    return data


def resolve_split_path(data_yaml: Path, data: dict, split: str) -> Path:
    value = data.get(split)
    if value is None:
        raise KeyError(f"Dataset yaml has no '{split}' field: {data_yaml}")
    path = Path(value)
    if not path.is_absolute():
        base = Path(data.get("path", data_yaml.parent))
        if not base.is_absolute():
            base = data_yaml.parent / base
        path = base / path
    return path.resolve()


def image_list(split_path: Path) -> list[Path]:
    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        root = split_path.parent
        images = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            img = Path(line)
            if not img.is_absolute():
                img = root / img
            images.append(img.resolve())
        return images
    if split_path.is_dir():
        return sorted(p.resolve() for p in split_path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"Split path not found: {split_path}")


def label_path_for_image(img: Path) -> Path:
    parts = list(img.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img.parent.parent / "labels" / img.parent.name / f"{img.stem}.txt"


def class_counts(images: list[Path]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for img in images:
        label_path = label_path_for_image(img)
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.strip().split()
            if len(fields) < 5:
                continue
            try:
                cls_id = int(float(fields[0]))
            except ValueError:
                continue
            counts[cls_id] = counts.get(cls_id, 0) + 1
    return counts


def apply_variant(src: Path, dst: Path, variant: str, args: argparse.Namespace) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if variant == "clean":
        shutil.copy2(src, dst)
        return

    with Image.open(src) as im:
        im = im.convert("RGB")
        if variant == "low_light":
            im = ImageEnhance.Brightness(im).enhance(args.low_light_factor)
            im.save(dst, quality=95)
        elif variant == "overexposure":
            im = ImageEnhance.Brightness(im).enhance(args.overexposure_factor)
            im = ImageEnhance.Contrast(im).enhance(1.15)
            im.save(dst, quality=95)
        elif variant == "gaussian_blur":
            im = im.filter(ImageFilter.GaussianBlur(radius=args.blur_radius))
            im.save(dst, quality=95)
        elif variant == "jpeg":
            im.save(dst, quality=args.jpeg_quality)
        else:
            raise ValueError(f"Unsupported variant: {variant}")


def prepare_variant_dataset(
    variant: str,
    images: list[Path],
    out_dir: Path,
    base_data: dict,
    args: argparse.Namespace,
) -> Path:
    root = out_dir / "datasets" / variant
    img_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    if root.exists():
        shutil.rmtree(root)
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for index, src_img in enumerate(images):
        suffix = src_img.suffix.lower()
        dst_name = f"{index:06d}_{src_img.stem}{suffix}"
        dst_img = img_dir / dst_name
        dst_label = label_dir / f"{Path(dst_name).stem}.txt"
        apply_variant(src_img, dst_img, variant, args)
        src_label = label_path_for_image(src_img)
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            dst_label.write_text("", encoding="utf-8")

    data = {
        "path": str(root),
        "train": "images/val",
        "val": "images/val",
        "names": base_data.get("names"),
    }
    if "nc" in base_data:
        data["nc"] = base_data["nc"]

    yaml_path = root / f"{variant}.yaml"
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yaml_path


def class_name(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, names.get(str(cls_id), cls_id)))
    if isinstance(names, list) and cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def collect_rows(variant: str, metrics, names, counts: dict[int, int]) -> list[dict]:
    rows = []
    box = metrics.box
    ap50_values = list(getattr(box, "ap50", []))
    ap_values = list(getattr(box, "ap", []))
    n_classes = max(len(ap_values), len(ap50_values), max(counts.keys(), default=-1) + 1)
    for cls_id in range(n_classes):
        rows.append(
            {
                "variant": variant,
                "class_id": cls_id,
                "class_name": class_name(names, cls_id),
                "instances": counts.get(cls_id, 0),
                "ap50": ap50_values[cls_id] if cls_id < len(ap50_values) else "",
                "ap50_95": ap_values[cls_id] if cls_id < len(ap_values) else "",
                "mean_precision": box.mp,
                "mean_recall": box.mr,
                "mean_map50": box.map50,
                "mean_map50_95": box.map,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    if not args.data.exists():
        raise FileNotFoundError(args.data)

    base_data = load_data_yaml(args.data)
    val_images = image_list(resolve_split_path(args.data, base_data, "val"))
    if args.max_images > 0:
        val_images = val_images[: args.max_images]
    if not val_images:
        raise ValueError("No validation images found.")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    out_dir = args.project / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    all_rows = []

    for variant in variants:
        yaml_path = prepare_variant_dataset(variant, val_images, out_dir, base_data, args)
        counts = class_counts(val_images)
        metrics = model.val(
            data=str(yaml_path),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(out_dir / "vals"),
            name=variant,
            plots=True,
            save_json=False,
            verbose=True,
        )
        all_rows.extend(collect_rows(variant, metrics, model.names, counts))

    csv_path = out_dir / "class_robustness_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved summary: {csv_path}")


if __name__ == "__main__":
    main()

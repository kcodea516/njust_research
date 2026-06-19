import argparse
import csv
import os
import shutil
from pathlib import Path

import yaml
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
    parser = argparse.ArgumentParser(description="Evaluate YOLO robustness by target-size groups.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained best.pt.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to dataset yaml.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="Vehicle5_size_robustness")
    parser.add_argument(
        "--group-mode",
        choices=("quantile", "threshold"),
        default="quantile",
        help="quantile creates balanced small/medium/large splits; threshold uses area thresholds.",
    )
    parser.add_argument("--small-thr", type=float, default=0.02, help="Normalized bbox-area threshold for small.")
    parser.add_argument("--medium-thr", type=float, default=0.10, help="Normalized bbox-area threshold for medium.")
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


def read_label_areas(label_path: Path) -> list[tuple[int, float]]:
    if not label_path.exists():
        return []
    items = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if len(fields) < 5:
            continue
        try:
            cls_id = int(float(fields[0]))
            width = float(fields[3])
            height = float(fields[4])
        except ValueError:
            continue
        items.append((cls_id, max(width * height, 0.0)))
    return items


def assign_groups(records: list[dict], mode: str, small_thr: float, medium_thr: float) -> dict[str, list[dict]]:
    groups = {"small": [], "medium": [], "large": []}
    if mode == "threshold":
        for record in records:
            area = record["max_area"]
            if area < small_thr:
                groups["small"].append(record)
            elif area < medium_thr:
                groups["medium"].append(record)
            else:
                groups["large"].append(record)
        return groups

    ordered = sorted(records, key=lambda r: r["max_area"])
    n = len(ordered)
    if n == 0:
        return groups
    cut1 = max(1, n // 3)
    cut2 = max(cut1 + 1, (2 * n) // 3)
    groups["small"] = ordered[:cut1]
    groups["medium"] = ordered[cut1:cut2]
    groups["large"] = ordered[cut2:]
    return groups


def write_list(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(r["image"]) for r in records) + "\n", encoding="utf-8")


def write_subset_yaml(path: Path, base_data: dict, val_list: Path) -> None:
    subset = dict(base_data)
    subset["val"] = str(val_list)
    path.write_text(yaml.safe_dump(subset, allow_unicode=True, sort_keys=False), encoding="utf-8")


def metric_row(group: str, records: list[dict], metrics) -> dict:
    box = metrics.box
    class_counts: dict[int, int] = {}
    for record in records:
        for cls_id, _ in record["labels"]:
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
    areas = [record["max_area"] for record in records]
    return {
        "group": group,
        "images": len(records),
        "instances": sum(class_counts.values()),
        "area_min": min(areas) if areas else "",
        "area_max": max(areas) if areas else "",
        "precision": box.mp,
        "recall": box.mr,
        "map50": box.map50,
        "map50_95": box.map,
        "class_instance_counts": ";".join(f"{k}:{v}" for k, v in sorted(class_counts.items())),
    }


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    if not args.data.exists():
        raise FileNotFoundError(args.data)

    data = load_data_yaml(args.data)
    val_images = image_list(resolve_split_path(args.data, data, "val"))
    records = []
    for img in val_images:
        labels = read_label_areas(label_path_for_image(img))
        if labels:
            records.append({"image": img, "labels": labels, "max_area": max(area for _, area in labels)})

    if not records:
        raise ValueError("No validation labels found. Check data yaml and labels directory.")

    out_dir = args.project / args.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    subset_dir = out_dir / "subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)

    groups = assign_groups(records, args.group_mode, args.small_thr, args.medium_thr)
    model = YOLO(str(args.weights))
    rows = []

    for group, group_records in groups.items():
        if not group_records:
            continue
        list_path = subset_dir / f"{group}.txt"
        yaml_path = subset_dir / f"{group}.yaml"
        write_list(list_path, group_records)
        write_subset_yaml(yaml_path, data, list_path)
        metrics = model.val(
            data=str(yaml_path),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(out_dir),
            name=group,
            plots=True,
            save_json=False,
            verbose=True,
        )
        rows.append(metric_row(group, group_records, metrics))

    csv_path = out_dir / "size_robustness_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary: {csv_path}")


if __name__ == "__main__":
    main()

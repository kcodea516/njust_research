from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/overload_weight_dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["under_25", "weight_25_40", "weight_40_52", "over_52"]


@dataclass(frozen=True)
class Sample:
    image_path: Path
    weight_ton: float
    source: str
    source_label: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a clean YOLO-compatible full-image overload weight-interval dataset."
    )
    parser.add_argument("--source", type=Path, required=True, help="Extracted overload source directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Clean dataset output directory.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bins",
        default="25,40,52",
        help="Weight thresholds in tons. Default creates <=25, (25,40], (40,52], >52.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to place images into the clean dataset.",
    )
    parser.add_argument("--image-col", default=None, help="Optional spreadsheet image filename/path column.")
    parser.add_argument("--weight-col", default=None, help="Optional spreadsheet weight column.")
    parser.add_argument(
        "--allow-empty-class",
        action="store_true",
        help="Allow generating a dataset even when some weight bands have zero samples.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    thresholds = parse_bins(args.bins)

    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"--source must be an extracted directory: {source}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Add --overwrite to rebuild it.")
        shutil.rmtree(output)

    image_index = build_image_index(source)
    samples = collect_samples(
        source=source,
        image_index=image_index,
        image_col=args.image_col,
        weight_col=args.weight_col,
    )
    if not samples:
        raise RuntimeError(
            "No usable overload samples found. Expected labels.txt, csv/xlsx table, "
            "or image filenames ending with a weight such as xxx_56.jpg."
        )

    unique_samples = deduplicate_samples(samples)
    split_map = stratified_split(unique_samples, thresholds, args.val_ratio, args.seed)
    write_dataset(output, source, unique_samples, split_map, thresholds, args.copy_mode)

    counts = count_by_class(unique_samples, thresholds)
    if not args.allow_empty_class:
        empty = [CLASS_NAMES[i] for i, count in enumerate(counts) if count == 0]
        if empty:
            raise RuntimeError(
                "Dataset was generated but some classes are empty: "
                + ", ".join(empty)
                + ". Use --allow-empty-class for a temporary baseline, or obtain the missing raw data/table."
            )

    print(f"Clean dataset written to: {output}")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"class {idx} {name}: {counts[idx]}")


def parse_bins(text: str) -> tuple[float, float, float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != 3:
        raise ValueError("--bins must contain exactly three thresholds, e.g. 25,40,52")
    if values != sorted(values):
        raise ValueError("--bins must be sorted from small to large")
    return values[0], values[1], values[2]


def is_junk_path(path: Path) -> bool:
    parts = set(path.parts)
    name = path.name
    return "__MACOSX" in parts or name == ".DS_Store" or name.startswith("._")


def build_image_index(source: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in source.rglob("*"):
        if is_junk_path(path) or not path.is_file():
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            index[path.name].append(path)
    return index


def collect_samples(
    source: Path,
    image_index: dict[str, list[Path]],
    image_col: str | None,
    weight_col: str | None,
) -> list[Sample]:
    samples: list[Sample] = []
    samples.extend(samples_from_tables(source, image_index, image_col, weight_col))
    samples.extend(samples_from_label_txt(source, image_index))

    covered_paths = {sample.image_path for sample in samples}
    for paths in image_index.values():
        for image_path in paths:
            if image_path in covered_paths:
                continue
            weight = parse_weight_from_filename(image_path.name)
            if weight is not None:
                samples.append(Sample(image_path=image_path, weight_ton=weight, source="filename"))
    return samples


def samples_from_label_txt(source: Path, image_index: dict[str, list[Path]]) -> list[Sample]:
    samples: list[Sample] = []
    for label_file in source.rglob("labels.txt"):
        if is_junk_path(label_file):
            continue
        for line in read_text_fallback(label_file).splitlines():
            text = line.strip()
            if not text or "," not in text:
                continue
            image_name, raw_label = [part.strip() for part in text.split(",", 1)]
            weight = parse_weight_from_filename(image_name)
            if weight is None:
                continue
            for image_path in find_image(image_name, image_index):
                samples.append(
                    Sample(
                        image_path=image_path,
                        weight_ton=weight,
                        source=str(label_file.relative_to(source)),
                        source_label=raw_label,
                    )
                )
    return samples


def samples_from_tables(
    source: Path,
    image_index: dict[str, list[Path]],
    image_col: str | None,
    weight_col: str | None,
) -> list[Sample]:
    samples: list[Sample] = []
    for table in source.rglob("*"):
        if is_junk_path(table) or not table.is_file():
            continue
        suffix = table.suffix.lower()
        if suffix == ".csv":
            rows = read_csv_rows(table)
        elif suffix in {".xlsx", ".xls"}:
            rows = read_excel_rows(table)
        else:
            continue
        if not rows:
            continue

        inferred_image_col = image_col or infer_column(rows[0], ["image", "img", "file", "filename", "path", "图片", "图像", "文件名", "照片"])
        inferred_weight_col = weight_col or infer_column(rows[0], ["weight", "ton", "tons", "gross", "重量", "总重", "车货总重", "称重", "吨"])
        if not inferred_image_col or not inferred_weight_col:
            continue

        for row in rows:
            image_value = str(row.get(inferred_image_col, "")).strip()
            weight_value = str(row.get(inferred_weight_col, "")).strip()
            weight = parse_number(weight_value)
            if not image_value or weight is None:
                continue
            for image_path in find_image(Path(image_value).name, image_index):
                samples.append(
                    Sample(
                        image_path=image_path,
                        weight_ton=weight,
                        source=str(table.relative_to(source)),
                        source_label=None,
                    )
                )
    return samples


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    text = read_text_fallback(path)
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel
    return list(csv.DictReader(text.splitlines(), dialect=dialect))


def read_excel_rows(path: Path) -> list[dict[str, object]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            f"Found spreadsheet {path}, but pandas is not installed. "
            "Install pandas/openpyxl or convert the table to CSV."
        ) from exc
    frame = pd.read_excel(path)
    return frame.fillna("").to_dict(orient="records")


def read_text_fallback(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def infer_column(row: dict[str, object], candidates: list[str]) -> str | None:
    normalized = {str(key).strip().lower(): str(key) for key in row.keys()}
    for candidate in candidates:
        needle = candidate.lower()
        for key_lower, original in normalized.items():
            if needle in key_lower:
                return original
    return None


def find_image(image_name: str, image_index: dict[str, list[Path]]) -> list[Path]:
    if image_name in image_index:
        return image_index[image_name]
    lower_map = {name.lower(): paths for name, paths in image_index.items()}
    return lower_map.get(image_name.lower(), [])


def parse_weight_from_filename(name: str) -> float | None:
    stem = Path(name).stem
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)$", stem)
    if not match:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)$", stem)
    return normalize_weight(float(match.group(1))) if match else None


def parse_number(value: str) -> float | None:
    match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", value.replace(",", ""))
    return normalize_weight(float(match.group(0))) if match else None


def normalize_weight(value: float) -> float:
    """Normalize weight to tons.

    Source filenames may end with either tons, such as ``_56.jpg``, or kilograms,
    such as ``_86800.jpg``. Values above 200 are treated as kilograms.
    """

    return value / 1000.0 if value > 200 else value


def class_id_for_weight(weight: float, thresholds: tuple[float, float, float]) -> int:
    first, second, third = thresholds
    if weight <= first:
        return 0
    if weight <= second:
        return 1
    if weight <= third:
        return 2
    return 3


def deduplicate_samples(samples: list[Sample]) -> list[Sample]:
    by_path: dict[Path, Sample] = {}
    for sample in samples:
        previous = by_path.get(sample.image_path)
        if previous is None or sample.source.endswith((".csv", ".xlsx", ".xls")):
            by_path[sample.image_path] = sample
    return list(by_path.values())


def stratified_split(
    samples: list[Sample],
    thresholds: tuple[float, float, float],
    val_ratio: float,
    seed: int,
) -> dict[Path, str]:
    rng = random.Random(seed)
    grouped: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[class_id_for_weight(sample.weight_ton, thresholds)].append(sample)

    split_map: dict[Path, str] = {}
    for class_samples in grouped.values():
        shuffled = list(class_samples)
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
        val_paths = {sample.image_path for sample in shuffled[:val_count]}
        for sample in shuffled:
            split_map[sample.image_path] = "val" if sample.image_path in val_paths else "train"
    return split_map


def write_dataset(
    output: Path,
    source: Path,
    samples: list[Sample],
    split_map: dict[Path, str],
    thresholds: tuple[float, float, float],
    copy_mode: str,
) -> None:
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    for index, sample in enumerate(samples):
        split = split_map[sample.image_path]
        cls_id = class_id_for_weight(sample.weight_ton, thresholds)
        dst_name = f"overload_{index:08d}{sample.image_path.suffix.lower()}"
        dst_image = output / "images" / split / dst_name
        dst_label = output / "labels" / split / f"{Path(dst_name).stem}.txt"
        place_file(sample.image_path, dst_image, copy_mode)
        dst_label.write_text(f"{cls_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")
        manifest_rows.append(
            {
                "split": split,
                "image": str(dst_image.relative_to(output)).replace("\\", "/"),
                "source_image": str(sample.image_path.relative_to(source)),
                "weight_ton": sample.weight_ton,
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "source": sample.source,
                "source_label": sample.source_label or "",
            }
        )

    write_manifest(output / "manifest.csv", manifest_rows)
    write_yaml(output / "overload.yaml", output)
    write_summary(output / "dataset_summary.json", manifest_rows, thresholds)


def place_file(src: Path, dst: Path, copy_mode: str) -> None:
    if copy_mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["split", "image", "source_image", "weight_ton", "class_id", "class_name", "source", "source_label"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(path: Path, output: Path) -> None:
    resolved_output = output.resolve()
    try:
        root = str(resolved_output.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        root = str(resolved_output).replace("\\", "/")
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    content = (
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n"
    )
    path.write_text(content, encoding="utf-8")


def write_summary(path: Path, rows: list[dict[str, object]], thresholds: tuple[float, float, float]) -> None:
    split_counts = Counter(row["split"] for row in rows)
    class_counts = Counter(int(row["class_id"]) for row in rows)
    summary = {
        "task": "visual_weight_interval_classification_yolo_full_image",
        "thresholds_ton": {
            "under_25": f"<= {thresholds[0]}",
            "weight_25_40": f"> {thresholds[0]} and <= {thresholds[1]}",
            "weight_40_52": f"> {thresholds[1]} and <= {thresholds[2]}",
            "over_52": f"> {thresholds[2]}",
        },
        "total": len(rows),
        "split_counts": dict(split_counts),
        "class_counts": {CLASS_NAMES[idx]: class_counts.get(idx, 0) for idx in range(len(CLASS_NAMES))},
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def count_by_class(samples: list[Sample], thresholds: tuple[float, float, float]) -> list[int]:
    counts = [0 for _ in CLASS_NAMES]
    for sample in samples:
        counts[class_id_for_weight(sample.weight_ton, thresholds)] += 1
    return counts


if __name__ == "__main__":
    main()

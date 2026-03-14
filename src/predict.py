"""
模型推理与可视化服务
====================
功能：
1. 对图片/视频进行批量推理
2. 生成带检测框的可视化结果
3. 导出检测结果为 JSON/CSV
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import draw_detections


def predict_images(
    model_path: str,
    input_dir: str,
    output_dir: str = None,
    conf_threshold: float = 0.25,
    save_visualizations: bool = True,
    save_json: bool = True,
    save_csv: bool = True
):
    """
    对目录中的所有图片进行批量推理
    
    Args:
        model_path: 模型权重路径
        input_dir: 输入图片目录
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        save_visualizations: 是否保存可视化结果
        save_json: 是否保存 JSON 格式结果
        save_csv: 是否保存 CSV 格式结果
    """
    input_path = Path(input_dir)
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "results" / "predictions")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect image files
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        img_files.extend(input_path.glob(ext))
    
    if not img_files:
        print(f"❌ 未在 {input_dir} 中找到图片")
        return
    
    print("=" * 60)
    print("  超限车辆智能感知系统 - 批量推理")
    print("=" * 60)
    print(f"  模型: {model_path}")
    print(f"  输入: {input_dir} ({len(img_files)} 张图片)")
    print(f"  输出: {output_dir}")
    print(f"  置信度阈值: {conf_threshold}")
    print("=" * 60)
    
    model = YOLO(model_path)
    all_results = []
    
    # Create visualization directory
    if save_visualizations:
        vis_dir = output_path / "visualizations"
        vis_dir.mkdir(exist_ok=True)
    
    for img_file in tqdm(img_files, desc="推理中"):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        # Run inference
        results = model.predict(img, conf=conf_threshold, verbose=False)
        
        detections = []
        for r in results:
            for box in r.boxes:
                det = {
                    "class_id": int(box.cls),
                    "class_name": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": [float(x) for x in box.xyxy[0].tolist()],
                }
                detections.append(det)
        
        result_entry = {
            "image": img_file.name,
            "image_path": str(img_file),
            "num_detections": len(detections),
            "detections": detections,
        }
        all_results.append(result_entry)
        
        # Save visualization
        if save_visualizations and detections:
            vis_img = draw_detections(img, detections)
            vis_path = vis_dir / f"det_{img_file.name}"
            cv2.imwrite(str(vis_path), vis_img)
    
    # Save JSON results
    if save_json:
        json_path = output_path / "predictions.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"📄 JSON 结果: {json_path}")
    
    # Save CSV results
    if save_csv:
        csv_path = output_path / "predictions.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "image", "detection_id", "class_id", "class_name",
                "confidence", "x1", "y1", "x2", "y2"
            ])
            for result in all_results:
                for i, det in enumerate(result["detections"]):
                    writer.writerow([
                        result["image"], i,
                        det["class_id"], det["class_name"],
                        f"{det['confidence']:.4f}",
                        *[f"{x:.1f}" for x in det["bbox"]]
                    ])
        print(f"📄 CSV 结果: {csv_path}")
    
    # Print summary
    total_detections = sum(r["num_detections"] for r in all_results)
    class_counts = {}
    for r in all_results:
        for det in r["detections"]:
            name = det["class_name"]
            class_counts[name] = class_counts.get(name, 0) + 1
    
    print(f"\n📊 推理结果统计:")
    print(f"  处理图片: {len(all_results)} 张")
    print(f"  检测目标: {total_detections} 个")
    if class_counts:
        print("  各类别数量:")
        for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {name}: {count}")
    
    return all_results


def predict_video(
    model_path: str,
    video_path: str,
    output_path: str = None,
    conf_threshold: float = 0.25,
    skip_frames: int = 0,
    max_frames: int = None
):
    """
    对视频进行推理
    
    Args:
        model_path: 模型权重路径
        video_path: 输入视频路径
        output_path: 输出视频路径
        conf_threshold: 置信度阈值
        skip_frames: 跳帧数（每隔N帧处理一次）
        max_frames: 最大处理帧数
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("=" * 60)
    print("  视频推理")
    print("=" * 60)
    print(f"  视频: {video_path}")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps}")
    print(f"  总帧数: {total_frames}")
    
    if output_path is None:
        output_path = str(
            Path(video_path).parent / f"det_{Path(video_path).name}"
        )
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detection_count = 0
    
    pbar = tqdm(total=min(total_frames, max_frames or total_frames))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames and frame_count >= max_frames:
            break
        
        # Skip frames if specified
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            out.write(frame)
            frame_count += 1
            pbar.update(1)
            continue
        
        # Run inference
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        
        # Draw detections
        detections = []
        for r in results:
            for box in r.boxes:
                det = {
                    "class_id": int(box.cls),
                    "class_name": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": [float(x) for x in box.xyxy[0].tolist()],
                }
                detections.append(det)
        
        detection_count += len(detections)
        annotated_frame = draw_detections(frame, detections)
        out.write(annotated_frame)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\n✅ 视频推理完成:")
    print(f"  处理帧数: {frame_count}")
    print(f"  检测目标: {detection_count}")
    print(f"  输出视频: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="超限车辆智能感知系统 - 推理与可视化"
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # images command
    img_parser = subparsers.add_parser("images", help="批量图片推理")
    img_parser.add_argument("--model", required=True, help="模型权重路径")
    img_parser.add_argument("--input-dir", required=True, help="输入图片目录")
    img_parser.add_argument("--output-dir", default=None, help="输出目录")
    img_parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    
    # video command
    vid_parser = subparsers.add_parser("video", help="视频推理")
    vid_parser.add_argument("--model", required=True, help="模型权重路径")
    vid_parser.add_argument("--video", required=True, help="输入视频路径")
    vid_parser.add_argument("--output", default=None, help="输出视频路径")
    vid_parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    vid_parser.add_argument("--skip-frames", type=int, default=0, help="跳帧数")
    vid_parser.add_argument("--max-frames", type=int, default=None, help="最大处理帧数")
    
    args = parser.parse_args()
    
    if args.command == "images":
        predict_images(
            args.model, args.input_dir,
            output_dir=args.output_dir,
            conf_threshold=args.conf,
        )
    elif args.command == "video":
        predict_video(
            args.model, args.video,
            output_path=args.output,
            conf_threshold=args.conf,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

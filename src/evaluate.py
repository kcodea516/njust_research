"""
模型评估脚本
==============
功能：
1. 在验证集/测试集上评估模型性能
2. 计算 mAP、Precision、Recall、F1-Score
3. 生成混淆矩阵
4. 生成 P-R 曲线
5. 推理速度测试（FPS）
6. 对比多个模型的性能
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_model(
    model_path: str,
    data_yaml: str,
    img_size: int = 640,
    batch_size: int = 8,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    save_dir: str = None,
    verbose: bool = True
) -> dict:
    """
    在验证集上评估模型性能
    
    Args:
        model_path: 模型权重路径 (.pt)
        data_yaml: 数据集配置文件路径
        img_size: 输入图像尺寸
        batch_size: 批次大小
        conf_threshold: 置信度阈值
        iou_threshold: IoU 阈值
        save_dir: 结果保存目录
        verbose: 是否输出详细信息
    
    Returns:
        dict: 包含所有评估指标的字典
    """
    if save_dir is None:
        save_dir = str(PROJECT_ROOT / "results" / "evaluation")
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  超限车辆智能感知系统 - 模型评估")
    print("=" * 60)
    print(f"  模型: {model_path}")
    print(f"  数据集: {data_yaml}")
    print(f"  置信度阈值: {conf_threshold}")
    print(f"  IoU阈值: {iou_threshold}")
    print("=" * 60)
    
    # Load model
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        conf=conf_threshold,
        iou=iou_threshold,
        save_json=True,
        save_dir=save_dir,
        plots=True,
        verbose=verbose,
    )
    
    # Extract metrics
    metrics = {
        "model": str(model_path),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "img_size": img_size,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
        },
        "metrics": {
            "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else None,
            "mAP50-95": float(results.box.map) if hasattr(results.box, 'map') else None,
            "precision": float(results.box.mp) if hasattr(results.box, 'mp') else None,
            "recall": float(results.box.mr) if hasattr(results.box, 'mr') else None,
        },
    }
    
    # Per-class metrics
    if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
        per_class = {}
        names = model.names
        for i, cls_idx in enumerate(results.box.ap_class_index):
            cls_name = names.get(int(cls_idx), f"class_{cls_idx}")
            per_class[cls_name] = {
                "ap50": float(results.box.ap50[i]) if i < len(results.box.ap50) else None,
                "ap": float(results.box.ap[i]) if i < len(results.box.ap) else None,
            }
        metrics["per_class"] = per_class
    
    # Save metrics
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n📊 评估结果:")
    print(f"  mAP@0.5:     {metrics['metrics'].get('mAP50', 'N/A')}")
    print(f"  mAP@0.5:0.95: {metrics['metrics'].get('mAP50-95', 'N/A')}")
    print(f"  Precision:   {metrics['metrics'].get('precision', 'N/A')}")
    print(f"  Recall:      {metrics['metrics'].get('recall', 'N/A')}")
    
    if "per_class" in metrics:
        print("\n  各类别 AP@0.5:")
        for cls_name, cls_metrics in metrics["per_class"].items():
            ap50 = cls_metrics.get("ap50", "N/A")
            if isinstance(ap50, float):
                ap50 = f"{ap50:.4f}"
            print(f"    {cls_name}: {ap50}")
    
    print(f"\n  📄 详细指标已保存至: {metrics_path}")
    
    return metrics


def test_inference_speed(
    model_path: str,
    img_size: int = 640,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = None
) -> dict:
    """
    测试模型推理速度 (FPS)
    
    Args:
        model_path: 模型权重路径
        img_size: 输入图像尺寸
        num_warmup: 预热迭代次数
        num_iterations: 实际测试迭代次数
        device: 推理设备
    
    Returns:
        dict: 推理速度报告
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"\n⏱️  推理速度测试 (device={device})")
    print(f"  预热: {num_warmup} 次, 测试: {num_iterations} 次")
    
    model = YOLO(model_path)
    
    # Create dummy input
    dummy_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    
    # Warmup
    print("  预热中...")
    for _ in range(num_warmup):
        model.predict(dummy_img, device=device, verbose=False)
    
    # Speed test
    print("  测试中...")
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        model.predict(dummy_img, device=device, verbose=False)
        end = time.perf_counter()
        times.append(end - start)
    
    times = np.array(times) * 1000  # Convert to ms
    
    speed_report = {
        "model": str(model_path),
        "device": device,
        "img_size": img_size,
        "num_iterations": num_iterations,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "fps": float(1000.0 / np.mean(times)),
    }
    
    print(f"\n  📊 推理速度结果:")
    print(f"    平均耗时: {speed_report['mean_ms']:.2f} ± {speed_report['std_ms']:.2f} ms")
    print(f"    FPS: {speed_report['fps']:.1f}")
    print(f"    最快/最慢: {speed_report['min_ms']:.2f} / {speed_report['max_ms']:.2f} ms")
    
    return speed_report


def compare_models(
    model_paths: list,
    data_yaml: str,
    save_dir: str = None,
    img_size: int = 640
) -> dict:
    """
    对比多个模型的性能
    
    Args:
        model_paths: 模型权重路径列表
        data_yaml: 数据集配置文件路径
        save_dir: 结果保存目录
        img_size: 输入图像尺寸
    
    Returns:
        dict: 对比结果
    """
    if save_dir is None:
        save_dir = str(PROJECT_ROOT / "results" / "comparison")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  模型对比评估")
    print("=" * 60)
    
    all_results = []
    
    for model_path in model_paths:
        print(f"\n{'─' * 40}")
        print(f"评估模型: {Path(model_path).name}")
        print(f"{'─' * 40}")
        
        # Evaluate accuracy
        metrics = evaluate_model(
            model_path, data_yaml,
            img_size=img_size,
            save_dir=os.path.join(save_dir, Path(model_path).stem)
        )
        
        # Test speed
        speed = test_inference_speed(model_path, img_size=img_size)
        
        result = {
            "model_name": Path(model_path).stem,
            "model_path": str(model_path),
            **metrics["metrics"],
            **speed
        }
        all_results.append(result)
    
    # Generate comparison chart
    _plot_comparison(all_results, save_dir)
    
    # Save comparison results
    comparison_path = os.path.join(save_dir, "comparison_results.json")
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("  📊 模型对比结果汇总")
    print("=" * 80)
    header = f"{'模型':<20} {'mAP@0.5':<12} {'mAP@0.5:0.95':<14} {'Precision':<12} {'FPS':<10}"
    print(header)
    print("-" * 68)
    for r in all_results:
        map50 = f"{r.get('mAP50', 0):.4f}" if r.get('mAP50') else "N/A"
        map5095 = f"{r.get('mAP50-95', 0):.4f}" if r.get('mAP50-95') else "N/A"
        prec = f"{r.get('precision', 0):.4f}" if r.get('precision') else "N/A"
        fps = f"{r.get('fps', 0):.1f}" if r.get('fps') else "N/A"
        print(f"{r['model_name']:<20} {map50:<12} {map5095:<14} {prec:<12} {fps:<10}")
    
    print(f"\n📄 对比结果已保存至: {comparison_path}")
    
    return all_results


def _plot_comparison(results: list, save_dir: str):
    """生成对比图表"""
    model_names = [r["model_name"] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("模型性能对比", fontsize=14, fontweight='bold')
    
    # mAP comparison
    map50_values = [r.get("mAP50", 0) or 0 for r in results]
    map5095_values = [r.get("mAP50-95", 0) or 0 for r in results]
    
    x = np.arange(len(model_names))
    width = 0.35
    axes[0].bar(x - width/2, map50_values, width, label='mAP@0.5', color='#2196F3')
    axes[0].bar(x + width/2, map5095_values, width, label='mAP@0.5:0.95', color='#FF9800')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].set_ylabel("mAP")
    axes[0].set_title("精度对比 (mAP)")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    
    # Precision & Recall
    precision_values = [r.get("precision", 0) or 0 for r in results]
    recall_values = [r.get("recall", 0) or 0 for r in results]
    axes[1].bar(x - width/2, precision_values, width, label='Precision', color='#4CAF50')
    axes[1].bar(x + width/2, recall_values, width, label='Recall', color='#E91E63')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].set_ylabel("Score")
    axes[1].set_title("Precision & Recall")
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)
    
    # FPS comparison
    fps_values = [r.get("fps", 0) or 0 for r in results]
    bars = axes[2].bar(model_names, fps_values, color='#9C27B0')
    axes[2].set_ylabel("FPS")
    axes[2].set_title("推理速度 (FPS)")
    axes[2].tick_params(axis='x', rotation=45)
    for bar, fps in zip(bars, fps_values):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{fps:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 对比图已保存至: {plot_path}")


def predict_single_image(
    model_path: str,
    image_path: str,
    conf_threshold: float = 0.25,
    save_path: str = None,
    show: bool = False
) -> dict:
    """
    对单张图片进行推理并可视化结果
    
    Args:
        model_path: 模型权重路径
        image_path: 图片路径
        conf_threshold: 置信度阈值
        save_path: 结果保存路径
        show: 是否显示结果
    
    Returns:
        dict: 检测结果
    """
    model = YOLO(model_path)
    results = model.predict(
        image_path,
        conf=conf_threshold,
        save=save_path is not None,
        save_dir=os.path.dirname(save_path) if save_path else None,
    )
    
    # Parse results
    detections = []
    for r in results:
        for box in r.boxes:
            det = {
                "class_id": int(box.cls),
                "class_name": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist(),
            }
            detections.append(det)
    
    print(f"\n🔍 检测结果 ({image_path}):")
    for det in detections:
        print(f"  [{det['class_name']}] 置信度: {det['confidence']:.3f} "
              f"位置: [{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, "
              f"{det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]")
    
    if not detections:
        print("  未检测到车辆")
    
    return {"image": image_path, "detections": detections}


def main():
    parser = argparse.ArgumentParser(
        description="超限车辆智能感知系统 - 模型评估脚本"
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # eval command
    eval_parser = subparsers.add_parser("eval", help="评估模型性能")
    eval_parser.add_argument("--model", required=True, help="模型权重路径")
    eval_parser.add_argument("--data", required=True, help="数据集配置文件")
    eval_parser.add_argument("--img-size", type=int, default=640)
    eval_parser.add_argument("--batch-size", type=int, default=8)
    eval_parser.add_argument("--conf", type=float, default=0.25)
    eval_parser.add_argument("--iou", type=float, default=0.45)
    eval_parser.add_argument("--save-dir", default=None)
    
    # speed command
    speed_parser = subparsers.add_parser("speed", help="测试推理速度")
    speed_parser.add_argument("--model", required=True, help="模型权重路径")
    speed_parser.add_argument("--img-size", type=int, default=640)
    speed_parser.add_argument("--iterations", type=int, default=100)
    
    # compare command
    compare_parser = subparsers.add_parser("compare", help="对比多个模型")
    compare_parser.add_argument("--models", nargs="+", required=True,
                               help="模型权重路径列表")
    compare_parser.add_argument("--data", required=True, help="数据集配置文件")
    compare_parser.add_argument("--save-dir", default=None)
    
    # predict command
    predict_parser = subparsers.add_parser("predict", help="单张图片推理")
    predict_parser.add_argument("--model", required=True, help="模型权重路径")
    predict_parser.add_argument("--image", required=True, help="图片路径")
    predict_parser.add_argument("--conf", type=float, default=0.25)
    predict_parser.add_argument("--save-path", default=None)
    
    args = parser.parse_args()
    
    if args.command == "eval":
        evaluate_model(
            args.model, args.data,
            img_size=args.img_size,
            batch_size=args.batch_size,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            save_dir=args.save_dir,
        )
    elif args.command == "speed":
        test_inference_speed(
            args.model,
            img_size=args.img_size,
            num_iterations=args.iterations,
        )
    elif args.command == "compare":
        compare_models(args.models, args.data, args.save_dir)
    elif args.command == "predict":
        predict_single_image(
            args.model, args.image,
            conf_threshold=args.conf,
            save_path=args.save_path,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

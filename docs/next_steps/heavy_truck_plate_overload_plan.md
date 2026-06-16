# Five-Class Heavy Truck, Plate Recognition, and Overload Plan

## 1. Current Decision

The next training task is the five-class heavy-truck detector. The current dataset is treated as fixed, so the first server task should be a clean baseline training run.

Focal loss should be treated as a second-stage experiment, not the first baseline. The baseline gives us the reference metrics needed to decide whether focal loss is actually helping. If the first model mainly fails on minority classes such as `Zhatu` or `Caoguan`, then we can compare focal loss against the baseline.

## 2. Five-Class Heavy-Truck Training

Dataset config:

```text
data/yolo_chexing_dataset/chexing.yaml
```

Classes:

```text
0 Kong
1 Sanhuo
2 Jizhuangxiang
3 Caoguan
4 Zhatu
```

Default initial weights:

```text
src/models/runs/detect/BIT_Vehicle_Model/weights/best.pt
```

This starts from the existing BIT vehicle detector and fine-tunes it on the five heavy-truck classes.

Recommended two-GPU command for the server:

```bash
conda activate ksy
python src/models/train_chexing.py \
  --data data/yolo_chexing_dataset/chexing.yaml \
  --weights src/models/runs/detect/BIT_Vehicle_Model/weights/best.pt \
  --epochs 100 \
  --batch 64 \
  --imgsz 640 \
  --device 0,1 \
  --workers 16 \
  --cache \
  --cos-lr \
  --name Heavy_Vehicle_Model_2x3090
```

If CUDA memory is still comfortable on two RTX 3090 cards, try `--batch 96`. If CUDA OOM appears, reduce to `--batch 48` or remove `--cache`.

## 3. Baseline Metrics to Read

After training, check:

- `mAP50`: whether boxes and classes are generally usable.
- `mAP50-95`: stricter detection quality across IoU thresholds.
- `precision`: how many predicted detections are correct.
- `recall`: how many real vehicles are found.
- per-class AP: especially `Caoguan` and `Zhatu`.
- confusion matrix: which truck types are confused with each other.
- PR curve and F1 curve: choose a practical confidence threshold.

Expected output directory:

```text
src/models/runs/detect/Heavy_Vehicle_Model_2x3090/
```

Important files:

```text
weights/best.pt
results.csv
confusion_matrix.png
PR_curve.png
F1_curve.png
```

Run validation after training:

```bash
python src/models/evaluate_chexing.py \
  --weights src/models/runs/detect/Heavy_Vehicle_Model_2x3090/weights/best.pt \
  --data data/yolo_chexing_dataset/chexing.yaml \
  --device 0 \
  --batch 64 \
  --name Heavy_Vehicle_Model_2x3090_val
```

## 4. Focal Loss Position

Focal loss is useful when easy majority-class examples dominate the loss and the model ignores hard or minority examples. For this project, it should be tried only after the baseline shows a concrete minority-class problem.

Experiment order:

1. Train baseline with current dataset.
2. Read per-class AP and confusion matrix.
3. If `Zhatu` or `Caoguan` recall/AP is poor, create a focal-loss branch.
4. Compare focal-loss run against the same baseline metrics.

Do not replace the baseline training script with custom focal loss before the first run.

## 5. Plate Recognition Plan

Use a two-stage plate pipeline:

1. Plate detection: detect the license-plate bounding box.
2. OCR recognition: crop the plate and recognize text.

Downloaded detector candidate:

```text
weights/license_plate_detector/best.pt
```

Source:

```text
Koushim/yolov8-license-plate-detection
```

This can be used as an initial plate detector. It may need fine-tuning on Chinese plate images because the highway-camera scene and Chinese plate appearance differ from many generic plate datasets.

Chinese plate data candidates for later:

```text
okita-souji/ccpd2019train
zenitsu09/ccpd-100k-yolo
zenitsu09/ccpd-ocr-recognition
```

Recommended OCR direction:

- Use PaddleOCR or a CRNN/Transformer OCR model for Chinese plate text.
- Keep OCR separate from plate detection.
- Do not fake plate numbers before a real OCR model is connected.

## 6. Final Integration Order

1. Vehicle detector: find vehicles and identify `Truck`.
2. Heavy-truck detector: run only when the primary vehicle is `Truck`.
3. Plate detector: detect plate region.
4. Plate OCR: recognize plate text from the crop.
5. Overload module: stay pending until there is a defensible model or rule source.

The API should continue returning separate blocks:

```text
detection
vehicle_type
plate
overload
workflow
```

This keeps the frontend stable while models are added one at a time.

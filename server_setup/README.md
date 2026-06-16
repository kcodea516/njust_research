# Server Setup

Create the conda environment:

```bash
cd /home/kang/research/server_setup
conda env create -f environment_ksy.yml
conda activate ksy
```

If the environment already exists:

```bash
conda activate ksy
pip install -r server_setup/requirements-ksy.txt
```

Check GPU availability:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

Train the five-class heavy-truck model on two RTX 3090 cards:

```bash
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

Validate the trained model:

```bash
python src/models/evaluate_chexing.py \
  --weights src/models/runs/detect/Heavy_Vehicle_Model_2x3090/weights/best.pt \
  --data data/yolo_chexing_dataset/chexing.yaml \
  --device 0 \
  --batch 64 \
  --name Heavy_Vehicle_Model_2x3090_val
```

Download the plate detector candidate if needed:

```bash
hf download Koushim/yolov8-license-plate-detection best.pt \
  --local-dir weights/license_plate_detector
```

# 高速道路重载货车超载嫌疑智能识别系统

本项目是面向本科软件著作权申报的交通执法辅助识别原型系统。系统支持上传高速道路监控或现场取证图片，调用已训练的 YOLO 目标检测模型识别车辆目标，并为后续超载嫌疑判定、车型细分类和车牌识别预留接口。

当前阶段已接入用户训练得到的目标检测权重：`/home/kang/research/src/models/runs/detect/BIT_Vehicle_Model/weights/best.pt`。超载嫌疑判定和车牌识别尚未接入真实模型，系统会明确显示“待接入/待复核”，不会伪造结论。项目采用 Vue + FastAPI 的一体化仓库结构，前端软件界面、后端 API 和模型调用模块放在同一个项目中，便于软著整理和后续扩展。

## 功能特性

- Vue 前端图片上传与预览
- 图片文件后端归档保存
- YOLO 目标检测模型调用
- 目标检测置信度阈值调节
- 检测框标注图生成
- 检测类别、置信度和坐标框展示
- 超载嫌疑识别预留接口
- 车牌号识别预留接口
- 模块接入状态结构化展示
- 面向交通管理部门的政务系统风格页面

## 项目结构

```text
truck-overload-detection-system/
├── backend/
│   ├── main.py
│   ├── models/
│   │   └── model_loader.py
│   ├── services/
│   │   ├── vehicle_detector.py
│   │   ├── overload_detector.py
│   │   ├── vehicle_type_classifier.py
│   │   └── plate_recognizer.py
│   ├── outputs/
│   ├── uploads/
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.vue
│       ├── main.js
│       ├── api/
│       │   └── vehicleAnalysis.js
│       ├── components/
│       │   └── ResultCard.vue
│       └── styles.css
└── docs/
    └── system_design.md
```

## 安装依赖

后端依赖：

```bash
cd /home/kang/research/truck-overload-detection-system/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果已经有可用 Python 环境，也可以直接安装依赖：

```bash
cd /home/kang/research/truck-overload-detection-system/backend
pip install -r requirements.txt
```

前端依赖：

```bash
cd /home/kang/research/truck-overload-detection-system/frontend
npm install
```

## 开发模式启动

先启动后端：

```bash
cd /home/kang/research/truck-overload-detection-system/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

再启动 Vue 前端：

```bash
cd /home/kang/research/truck-overload-detection-system/frontend
npm run dev
```

开发模式访问：

```text
http://127.0.0.1:5173
```

## 一体化部署启动

构建 Vue 前端：

```bash
cd /home/kang/research/truck-overload-detection-system/frontend
npm run build
```

启动 FastAPI 后端，后端会自动托管 `frontend/dist`：

```bash
cd /home/kang/research/truck-overload-detection-system/backend
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001
```

一体化访问：

```text
http://127.0.0.1:8001
```

接口文档：

```text
http://127.0.0.1:8001/docs
```

## API 示例

图片分析接口：

```text
POST /api/analyze
Content-Type: multipart/form-data
```

返回示例：

```json
{
  "detection": {
    "module_status": "connected",
    "module_message": "已接入 YOLO 目标检测模型",
    "detections": [
      {
        "id": 1,
        "class_name": "Truck",
        "class_label": "货车",
        "confidence": 0.91,
        "bbox": [12.5, 34.2, 420.8, 310.6]
      }
    ],
    "primary_vehicle": {
      "id": 1,
      "class_name": "Truck",
      "class_label": "货车",
      "confidence": 0.91,
      "bbox": [12.5, 34.2, 420.8, 310.6]
    },
    "annotated_image_url": "/outputs/example_detected.jpg"
  },
  "vehicle_type": {
    "vehicle_type": "货车",
    "vehicle_type_confidence": 0.91,
    "module_status": "derived",
    "module_message": "车型来自目标检测模型的类别输出，并非独立车型细分类模型"
  },
  "overload": {
    "overload_suspected": null,
    "overload_confidence": null,
    "risk_level": "待复核",
    "module_status": "pending",
    "module_message": "超载嫌疑识别模型尚未接入"
  },
  "plate": {
    "plate_number": null,
    "plate_confidence": null,
    "module_status": "pending",
    "module_message": "车牌检测与 OCR 识别模型尚未接入"
  },
  "image_filename": "20260518_103000_xxx.jpg",
  "processed_at": "2026-05-18T10:30:00"
}
```

## 非数据集图片为什么可能检不出来

目标检测模型不是只能识别训练集里的原图，而是学习训练数据中的视觉特征后，对新图片进行泛化推理。但泛化效果受训练数据分布影响明显：

- 当前模型类别来自 BIT-Vehicle 数据集：`Bus`、`Microbus`、`Minivan`、`Sedan`、`SUV`、`Truck`；
- 如果新图片视角、清晰度、车辆尺寸、遮挡情况、拍摄高度和训练集差异很大，置信度可能下降；
- 当前系统默认检测阈值为 `0.50`，低于阈值的候选框会被过滤；
- 可以在前端降低阈值提高召回率，但误检也会增加；
- 当前模型只做车辆目标检测，不负责判断超载、识别车牌或识别训练类别外的特殊车辆。

## 后续接入真实模型

模型替换位置：

- `backend/services/vehicle_detector.py`
- `backend/services/overload_detector.py`
- `backend/services/vehicle_type_classifier.py`
- `backend/services/plate_recognizer.py`
- `backend/models/model_loader.py`

建议保持 `/api/analyze` 的顶层结构不变，即 `detection`、`vehicle_type`、`overload`、`plate` 四类结果分开返回。这样后续接入新模型时，前端只需要读取对应模块状态和字段。

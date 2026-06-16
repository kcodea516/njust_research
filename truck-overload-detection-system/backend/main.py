from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from services.vehicle_detector import VehicleDetector
from services.vehicle_type_classifier import VehicleTypeClassifier


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
ULTRALYTICS_CONFIG_DIR = BASE_DIR / ".ultralytics"
FRONTEND_DIR = PROJECT_DIR / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="高速道路车辆智能识别检测系统",
    description="面向交通管理场景的图片上传、车辆目标检测与车型识别原型系统。",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vehicle_detector = VehicleDetector()
vehicle_type_classifier = VehicleTypeClassifier()


def build_pending_plate_result() -> dict:
    return {
        "module_status": "pending",
        "module_message": "车牌识别模型暂未接入",
        "plate_number": None,
        "plate_confidence": None,
    }


def build_pending_overload_result() -> dict:
    return {
        "module_status": "pending",
        "module_message": "超载检测模型暂未接入",
        "overload_suspected": None,
        "overload_confidence": None,
        "limit_weight": None,
        "actual_weight": None,
        "overload_ratio": None,
        "risk_level": None,
        "basis": "当前阶段仅展示超载检测模块骨架，尚未输出判定数据。",
    }


if FRONTEND_DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST_DIR / "assets"), name="assets")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/")
def index() -> FileResponse:
    if FRONTEND_DIST_DIR.exists():
        return FileResponse(FRONTEND_DIST_DIR / "index.html")
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok", "service": "vehicle-detection"}


@app.post("/api/analyze")
async def analyze_vehicle_image(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.5),
) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")
    if conf_threshold < 0.05 or conf_threshold > 0.95:
        raise HTTPException(status_code=400, detail="检测阈值需在 0.05 到 0.95 之间")

    original_filename = file.filename or "upload.jpg"
    suffix = Path(original_filename).suffix.lower() or ".jpg"
    saved_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex}{suffix}"
    image_path = UPLOAD_DIR / saved_name

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")

    image_path.write_bytes(content)

    detection_result = vehicle_detector.predict(image_path, OUTPUT_DIR, conf_threshold=conf_threshold)
    vehicle_type_result = vehicle_type_classifier.predict(image_path, detection_result.get("primary_vehicle"))

    return {
        "detection": detection_result,
        "vehicle_type": vehicle_type_result,
        "plate": build_pending_plate_result(),
        "overload": build_pending_overload_result(),
        "workflow": {
            "summary": "当前已接入车辆目标检测与车型识别模型；车牌识别和超载检测仍等待真实模型接入。",
            "next_steps": ["接入车牌检测与 OCR 模型", "接入超载嫌疑判定模型", "增加人工复核与历史记录"],
        },
        "image_filename": saved_name,
        "original_filename": original_filename,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }

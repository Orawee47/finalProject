# app.py (production-ready + API Key protected, Render-ready)
#
# Features:
# - API Key protection via header: x-api-key
# - Standard success/error responses
# - Upload size limit
# - Resize for faster inference + scales boxes back to original image size
# - /health and /classes endpoints
#
# Endpoints:
#   GET  /        -> info
#   GET  /health  -> health check
#   GET  /classes -> class list
#   POST /predict -> multipart/form-data field "file"
#
# Required header for /predict:
#   x-api-key: <your API_KEY>
#
# Env vars (Render):
#   API_KEY=<secret>
#
# Note:
# - For now CORS is open for testing (ALLOW_ALL_ORIGINS_FOR_NOW=True).
#   When you connect Netlify, lock it down.

import io
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Request,
    Header,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = "yolov8s(lr_4(aug)).pt"

# Class names from your data.yaml (ORDER MUST MATCH)
CLASS_NAMES = [
    "Alovera",
    "cucumber",
    "Galanga",
    "Garlic",
    "horapa",
    "Houttuynia_cordata",
    "Ivy_Gourd",
    "khaproa",
    "Mangosteen_Peel",
    "pluleaf",
    "Snake_Plant",
    "Turmeric",
]

# Security
API_KEY = os.getenv("API_KEY")  # must be set on Render

# For now (not connecting Netlify yet): allow all origins for testing
ALLOW_ALL_ORIGINS_FOR_NOW = True

# Upload & processing limits
MAX_UPLOAD_MB = 5          # max upload size (MB)
MAX_IMAGE_SIDE = 1280      # resize longest side to this (for speed)

# =========================
# APP INIT
# =========================
app = FastAPI(title="YOLOv8s SkinHerb API", version="3.0.0")

if ALLOW_ALL_ORIGINS_FOR_NOW:
    # If allow_origins=["*"], credentials must be False
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# =========================
# MODEL LOAD (once)
# =========================
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model from '{MODEL_PATH}': {e}")

# =========================
# HELPERS
# =========================
def get_class_name(cls_id: int) -> str:
    if 0 <= cls_id < len(CLASS_NAMES):
        return CLASS_NAMES[cls_id]
    return f"unknown_{cls_id}"


def standard_error(message: str, code: str = "BAD_REQUEST", status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={"ok": False, "error": {"code": code, "message": message}},
    )


def read_image_to_rgb_pil(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")


def resize_keep_aspect(pil_img: Image.Image, max_side: int) -> Image.Image:
    w, h = pil_img.size
    longest = max(w, h)
    if longest <= max_side:
        return pil_img
    scale = max_side / float(longest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return pil_img.resize((new_w, new_h), Image.BILINEAR)


def require_api_key(x_api_key: str = Header(default="")):
    """
    Require x-api-key header to match API_KEY env var.
    This protects /predict from public abuse.
    """
    if not API_KEY:
        # Misconfiguration on server
        raise HTTPException(status_code=500, detail="API_KEY not set on server")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# =========================
# GLOBAL ERROR FORMATTER
# =========================
@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else "Request error"
    code = "UNAUTHORIZED" if exc.status_code == 401 else "HTTP_EXCEPTION"
    return standard_error(detail, code=code, status_code=exc.status_code)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, __: Exception):
    return standard_error("Internal server error", code="INTERNAL_ERROR", status_code=500)


# =========================
# ROUTES
# =========================
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "message": "YOLOv8s API is running",
        "model": "YOLOv8s",
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "limits": {
            "max_upload_mb": MAX_UPLOAD_MB,
            "max_image_side": MAX_IMAGE_SIDE,
        },
        "security": {
            "api_key_required_for_predict": True,
        },
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "status": "healthy"}


@app.get("/classes")
def classes() -> Dict[str, Any]:
    return {
        "ok": True,
        "num_classes": len(CLASS_NAMES),
        "classes": [{"class_id": i, "class_name": n} for i, n in enumerate(CLASS_NAMES)],
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    _: Any = Depends(require_api_key),  # ✅ API key required
) -> Dict[str, Any]:
    # Content-type guard
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (jpg/png/webp).")

    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Size limit
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(img_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed is {MAX_UPLOAD_MB} MB.",
        )

    # Read original
    pil = read_image_to_rgb_pil(img_bytes)
    orig_w, orig_h = pil.size

    # Resize for speed
    processed = resize_keep_aspect(pil, MAX_IMAGE_SIDE)
    proc_w, proc_h = processed.size
    img_np = np.array(processed)  # RGB numpy

    # Inference
    start = time.time()
    results = model.predict(img_np, verbose=False)
    end = time.time()

    r = results[0]
    detections: List[Dict[str, Any]] = []

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy()

        # Scale boxes back to ORIGINAL image coordinates
        scale_x = orig_w / float(proc_w)
        scale_y = orig_h / float(proc_h)

        for box, conf, cls_id in zip(boxes, confs, clss):
            cls_id_int = int(cls_id)

            x1 = int(round(float(box[0]) * scale_x))
            y1 = int(round(float(box[1]) * scale_y))
            x2 = int(round(float(box[2]) * scale_x))
            y2 = int(round(float(box[3]) * scale_y))

            detections.append(
                {
                    "class_id": cls_id_int,
                    "class_name": get_class_name(cls_id_int),
                    "confidence": round(float(conf), 4),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

    # Top prediction (highest confidence) - useful for UI
    top_prediction: Optional[Dict[str, Any]] = None
    if detections:
        best = max(detections, key=lambda d: d["confidence"])
        top_prediction = {
            "class_id": best["class_id"],
            "class_name": best["class_name"],
            "confidence": best["confidence"],
        }

    return {
        "ok": True,
        "model": "YOLOv8s",
        "image": {"width": orig_w, "height": orig_h},
        "processed_image": {"width": proc_w, "height": proc_h},
        "inference_ms": int((end - start) * 1000),
        "top_prediction": top_prediction,
        "detections": detections,
    }

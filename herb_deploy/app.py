# app.py (FULL VERSION) - Render + Netlify-ready, API Key protected
#
# ✅ Supports multiple origins (Render test + Netlify prod + Netlify preview)
# ✅ API Key protection for /predict via header: x-api-key
# ✅ Standard JSON error format
# ✅ Upload size limit
# ✅ Resize for faster inference + scale boxes back to original image size
# ✅ Adds `label_key` for stable DB lookup (fixes "not found in database" mismatch)
# ✅ Optional simple HTML test page at GET /predict
#
# Env vars (Render):
#   API_KEY=<secret>                               (required)
#   MODEL_PATH=best.pt                             (optional, default best.pt)
#   MAX_UPLOAD_MB=5                                (optional)
#   MAX_IMAGE_SIDE=1280                            (optional)
#   ALLOWED_ORIGINS=https://xxx.netlify.app,https://yyy.onrender.com   (optional)
#   ALLOWED_ORIGIN_REGEX=https://.*\\.netlify\\.app                    (optional, for Netlify preview)
#
# Endpoints:
#   GET  /        -> info
#   GET  /health  -> health check + env status
#   GET  /classes -> class list
#   GET  /predict -> HTML test page (optional)
#   POST /predict -> multipart/form-data field "file" + header x-api-key

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
from fastapi.responses import JSONResponse, HTMLResponse
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")

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

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1280"))

# CORS allowlist (comma-separated)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX", "")  # e.g. https://.*\\.netlify\\.app

ORIGINS = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
if not ORIGINS:
    # dev fallback
    ORIGINS = ["http://localhost:3000", "http://localhost:5173"]

# =========================
# APP INIT
# =========================
app = FastAPI(title="YOLO SkinHerb API", version="3.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX or None,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],  # allow x-api-key
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


def to_label_key(name: str) -> str:
    # stable key for DB lookup
    return name.strip().lower().replace(" ", "_").replace("-", "_")


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


def require_api_key(x_api_key: str = Header(default="", alias="x-api-key")):
    """
    Require x-api-key header to match API_KEY env var.
    Protects /predict from public abuse.
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API_KEY not set on server")
    if x_api_key != api_key:
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
        "message": "YOLO API is running",
        "model_path": MODEL_PATH,
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "limits": {
            "max_upload_mb": MAX_UPLOAD_MB,
            "max_image_side": MAX_IMAGE_SIDE,
        },
        "security": {"api_key_required_for_predict": True},
        "cors": {
            "allowed_origins": ORIGINS,
            "allowed_origin_regex": ALLOWED_ORIGIN_REGEX or None,
        },
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "status": "healthy",
        "api_key_set": bool(os.getenv("API_KEY")),
        "allowed_origins_set": bool(os.getenv("ALLOWED_ORIGINS")),
        "allowed_origin_regex_set": bool(os.getenv("ALLOWED_ORIGIN_REGEX")),
    }


@app.get("/classes")
def classes() -> Dict[str, Any]:
    return {
        "ok": True,
        "num_classes": len(CLASS_NAMES),
        "classes": [{"class_id": i, "class_name": n, "label_key": to_label_key(n)} for i, n in enumerate(CLASS_NAMES)],
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    _: Any = Depends(require_api_key),
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
            class_name = get_class_name(cls_id_int)

            x1 = int(round(float(box[0]) * scale_x))
            y1 = int(round(float(box[1]) * scale_y))
            x2 = int(round(float(box[2]) * scale_x))
            y2 = int(round(float(box[3]) * scale_y))

            detections.append(
                {
                    "class_id": cls_id_int,
                    "class_name": class_name,
                    "label_key": to_label_key(class_name),  # ✅ stable DB lookup key
                    "confidence": round(float(conf), 4),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

    # Top prediction (highest confidence)
    top_prediction: Optional[Dict[str, Any]] = None
    if detections:
        best = max(detections, key=lambda d: d["confidence"])
        top_prediction = {
            "class_id": best["class_id"],
            "class_name": best["class_name"],
            "label_key": best["label_key"],  # ✅ use this to query DB
            "confidence": best["confidence"],
        }

    return {
        "ok": True,
        "model": "YOLO",
        "image": {"width": orig_w, "height": orig_h},
        "processed_image": {"width": proc_w, "height": proc_h},
        "inference_ms": int((end - start) * 1000),
        "top_prediction": top_prediction,
        "detections": detections,
    }


# Optional: Simple HTML test page (helps when testing without Swagger/Postman)
@app.get("/predict", response_class=HTMLResponse)
def predict_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>YOLO Predict Test</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 680px; margin: 40px auto; }
    input, button { margin-top: 10px; }
    pre { background: #f4f4f4; padding: 10px; overflow-x: auto; }
    .row { margin-bottom: 12px; }
  </style>
</head>
<body>
  <h2>YOLO /predict – Test Page</h2>
  <p>Upload an image and call POST /predict with x-api-key.</p>

  <div class="row">
    <label>API Key</label><br/>
    <input type="text" id="apiKey" placeholder="Enter API Key" style="width:100%;" />
  </div>

  <div class="row">
    <label>Select image</label><br/>
    <input type="file" id="fileInput" accept="image/*" />
  </div>

  <button onclick="send()">Predict</button>

  <h3>Response</h3>
  <pre id="output">-</pre>

  <script>
    async function send() {
      const apiKey = document.getElementById("apiKey").value;
      const fileInput = document.getElementById("fileInput");
      const output = document.getElementById("output");

      if (!apiKey) { alert("Please enter API key"); return; }
      if (!fileInput.files.length) { alert("Please select an image"); return; }

      const formData = new FormData();
      formData.append("file", fileInput.files[0]);

      output.textContent = "Sending...";

      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "x-api-key": apiKey },
          body: formData
        });

        const text = await res.text();
        output.textContent = text;
      } catch (err) {
        output.textContent = err.toString();
      }
    }
  </script>
</body>
</html>
"""

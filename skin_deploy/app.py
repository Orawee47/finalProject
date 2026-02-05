from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import threading
import gc
import time

import torch
import torchvision.transforms as T

from model import load_model

app = FastAPI(title="Skin Mask R-CNN API", version="1.0")

CONF_THRES = float(os.environ.get("CONF_THRES", "0.5"))
MAX_SIDE = int(os.environ.get("MAX_SIDE", "512"))      # ลดไว้ก่อน กัน OOM
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "8"))

CLASS_TH = {
    0: "พื้นหลัง",
    1: "สิว",
    2: "งูสวัด",
    3: "เริม",
    4: "สะเก็ดเงิน",
    5: "กลาก/เกลื้อน",
    6: "ลมพิษ",
    7: "ด่างขาว",
}
CLASS_EN = {
    0: "__background__",
    1: "Acne",
    2: "Herpes_Zoster",
    3: "Herpes_simplex",
    4: "Psoriasis",
    5: "Tinea",
    6: "Urticaria_Hives",
    7: "Vitiligo",
}

# Render free → CPU
device = "cpu"

transform = T.ToTensor()

_model = None
_model_err = None

# =========================
# CORS CONFIG
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def resize_max_side(img: Image.Image, max_side: int):
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh), Image.BILINEAR)


def _load_model_bg():
    global _model, _model_err
    try:
        t0 = time.time()
        m = load_model(device=device, num_classes=NUM_CLASSES)

        # warmup แบบที่ torchvision detection ต้องการ: list[tensor]
        dummy = [torch.zeros(3, 256, 256, device=device)]
        with torch.inference_mode():
            _ = m(dummy)

        _model = m
        _model_err = None
        print(f"✅ Model loaded in {time.time()-t0:.2f}s | device={device}")
    except Exception as e:
        _model = None
        _model_err = repr(e)
        print("❌ Startup model load failed:", _model_err)
    finally:
        gc.collect()


@app.on_event("startup")
def startup():
    # ให้ uvicorn เปิดพอร์ตก่อน แล้วค่อยโหลดโมเดลใน background
    threading.Thread(target=_load_model_bg, daemon=True).start()


def get_model():
    if _model is None:
        detail = "Model not ready"
        if _model_err:
            detail += f" (startup failed: {_model_err})"
        raise HTTPException(status_code=503, detail=detail)
    return _model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model_ready": _model is not None,
        "error": _model_err,
        "max_side": MAX_SIDE,
        "conf_thres": CONF_THRES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    m = get_model()

    # read image
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {repr(e)}")

    # resize ลด RAM
    img = resize_max_side(img, MAX_SIDE)

    # torchvision Mask R-CNN ต้องเป็น List[Tensor]
    x = transform(img).to(device)   # [C,H,W]
    inputs = [x]

    try:
        with torch.inference_mode():
            out = m(inputs)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {repr(e)}")
    finally:
        del inputs, x
        gc.collect()

    scores = out.get("scores", torch.empty((0,))).detach().cpu()
    labels = out.get("labels", torch.empty((0,), dtype=torch.int64)).detach().cpu()

    if scores.numel() == 0:
        return {"top1": None, "message": "No detections"}

    keep = scores >= CONF_THRES
    scores = scores[keep]
    labels = labels[keep]

    if scores.numel() == 0:
        return {"top1": None, "message": f"No detections >= conf {CONF_THRES}"}

    best_i = int(torch.argmax(scores).item())
    best_score = float(scores[best_i].item())
    best_label = int(labels[best_i].item())

    return {
        "top1": {
            "label_id": best_label,
            "label_en": CLASS_EN.get(best_label, f"Unknown_{best_label}"),
            "label_th": CLASS_TH.get(best_label, f"ไม่ทราบ_{best_label}"),
            "score": best_score,
        }
    }

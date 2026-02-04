# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
import io
import os
import threading
import gc
import time

import torch
import torchvision.transforms as T

from model import load_model

app = FastAPI(title="Skin Mask R-CNN API", version="1.0")

# -------------------------
# CONFIG
# -------------------------
CONF_THRES = float(os.environ.get("CONF_THRES", "0.5"))
MAX_SIDE = int(os.environ.get("MAX_SIDE", "384"))          # ลดภาพก่อน infer กัน OOM
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "8"))
WARMUP_SIDE = int(os.environ.get("WARMUP_SIDE", "128"))    # ลด warmup กัน spike
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", "1"))

# -------------------------
# LABELS
# -------------------------
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

# -------------------------
# DEVICE: Render free = CPU
# -------------------------
device = "cpu"

# ToTensor -> float [0..1] shape [C,H,W] (ใช้ได้กับ Mask R-CNN)
transform = T.ToTensor()

# -------------------------
# GLOBAL STATE
# -------------------------
_model = None
_model_err = None
_model_lock = threading.Lock()


def resize_max_side(img: Image.Image, max_side: int):
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh), Image.BILINEAR)


def _load_model_bg():
    """
    โหลดโมเดลใน background เพื่อให้ Render "เห็น port" ก่อน (ลดปัญหา port scan)
    และลด OOM/Spike ด้วย:
    - จำกัด torch threads
    - warmup ด้วยรูปเล็ก (list[tensor])
    """
    global _model, _model_err
    t0 = time.time()

    try:
        torch.set_num_threads(TORCH_THREADS)

        m = load_model(device=device, num_classes=NUM_CLASSES)

        # ✅ warmup แบบ detection: list[tensor] และให้เล็กที่สุด
        dummy = [torch.zeros(3, WARMUP_SIDE, WARMUP_SIDE, device=device)]
        with torch.inference_mode():
            _ = m(dummy)

        with _model_lock:
            _model = m
            _model_err = None

        print(f"✅ Model loaded & warmed up on {device} in {time.time()-t0:.2f}s")
    except Exception as e:
        with _model_lock:
            _model = None
            _model_err = repr(e)
        print("❌ Model load failed:", _model_err)
    finally:
        gc.collect()


@app.on_event("startup")
def startup():
    # ✅ เปิดพอร์ตก่อน แล้วค่อยโหลดโมเดลใน background
    t = threading.Thread(target=_load_model_bg, daemon=True)
    t.start()


def get_model():
    with _model_lock:
        m = _model
        err = _model_err
    if m is None:
        detail = "Model not ready"
        if err:
            detail += f" (startup failed: {err})"
        raise HTTPException(status_code=503, detail=detail)
    return m


@app.get("/health")
def health_check():
    with _model_lock:
        ready = _model is not None
        err = _model_err
    return {
        "status": "ok",
        "service": "skin-maskrcnn-api",
        "device": device,
        "model_ready": ready,
        "error": err,
        "max_side": MAX_SIDE,
        "conf_thres": CONF_THRES,
        "warmup_side": WARMUP_SIDE,
        "torch_threads": TORCH_THREADS,
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

    # resize ก่อน infer ลด RAM
    img = resize_max_side(img, MAX_SIDE)

    # Mask R-CNN ต้องเป็น List[Tensor]
    x = transform(img)  # [C,H,W] float32
    inputs = [x.to(device)]

    try:
        with torch.inference_mode():
            out = m(inputs)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {repr(e)}")
    finally:
        # ลด peak mem
        try:
            del inputs, x
        except Exception:
            pass
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

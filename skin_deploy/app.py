from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
import io
import os
import torch
import torchvision.transforms as T
from model import load_model

# ============================================================
# 0) ลด RAM / ลด spike (สำคัญบน Render free 512MB)
# ============================================================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

torch.set_num_threads(1)

# ============================================================
# 1) Config
# ============================================================
app = FastAPI(title="Skin Mask R-CNN API", version="1.0")

CONF_THRES = float(os.environ.get("CONF_THRES", "0.5"))

# Render free -> แนะนำ 640 หรือ 512 ถ้ายัง OOM
MAX_SIDE = int(os.environ.get("MAX_SIDE", "640"))

# บังคับ CPU บน Render (ลดปัญหา + RAM spike)
FORCE_CPU = os.environ.get("FORCE_CPU", "1")  # default=1
device = "cpu" if FORCE_CPU == "1" else ("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2) label map
# ============================================================
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

# ============================================================
# 3) Image preprocess
#    - resize ก่อน ToTensor เพื่อลด RAM / ลดโอกาส OOM
# ============================================================
transform = T.ToTensor()

def resize_max_side(img: Image.Image, max_side: int):
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh), Image.BILINEAR)

# ============================================================
# 4) Model lifecycle
# ============================================================
model = None

def _warmup_maskrcnn(m, device: str):
    """
    Mask R-CNN (torchvision) ตอน inference ต้องรับเป็น list[Tensor]
    """
    m.eval()
    dummy = torch.zeros(3, 256, 256, device=device)  # [C,H,W]
    with torch.inference_mode():
        _ = m([dummy])  # ✅ list[Tensor]

@app.on_event("startup")
def _startup_load_model():
    """
    โหลดตอนเริ่ม เพื่อให้ /predict ไม่ cold-start แล้ว timeout
    ถ้า OOM ตอน startup ให้ลด MAX_SIDE หรือไป lazy load
    """
    global model
    try:
        # ✅ โหลดด้วย device ที่กำหนด (แนะนำ cpu บน Render)
        model = load_model(device=device)
        model.eval()

        # ✅ warmup ให้ถูกกับ Mask R-CNN
        _warmup_maskrcnn(model, device)

        print(f"✅ Model loaded & warmed up on device={device} | MAX_SIDE={MAX_SIDE} | CONF_THRES={CONF_THRES}")
    except Exception as e:
        print("❌ Startup model load failed:", repr(e))
        model = None

def get_model():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready (startup failed)")
    return model

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "skin-maskrcnn-api",
        "device": device,
        "model_ready": model is not None,
        "max_side": MAX_SIDE,
        "conf_thres": CONF_THRES,
    }

# ============================================================
# 5) Predict endpoint
# ============================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    m = get_model()

    # ---- read image safely ----
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {repr(e)}")

    # ✅ resize ก่อน ToTensor
    img = resize_max_side(img, MAX_SIDE)

    # torchvision Mask R-CNN: inference expects list[Tensor]
    x = transform(img).to(device)  # [C,H,W]
    inputs = [x]

    try:
        with torch.inference_mode():
            out = m(inputs)[0]  # dict
    except RuntimeError as e:
        # เผื่อเจอ OOM/RuntimeError
        msg = repr(e)
        if "out of memory" in msg.lower():
            raise HTTPException(status_code=500, detail=f"OOM during inference. Try lower MAX_SIDE. err={msg}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {msg}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {repr(e)}")

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

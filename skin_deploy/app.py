from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
import io
import torch
import torchvision.transforms as T
from model import load_model

app = FastAPI(title="Skin Mask R-CNN API", version="1.0")

CONF_THRES = 0.5

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

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = T.ToTensor()

model = None

@app.on_event("startup")
def _startup_load_model():
    """โหลดโมเดลตอน start service เพื่อตัดปัญหา cold-start ยิงแล้ว 502"""
    global model
    try:
        model = load_model(device=device)
        model.eval()
        # warmup 1 ครั้ง (ช่วยลด spike ตอน request แรก)
        dummy = torch.zeros(1, 3, 256, 256, device=device)
        with torch.inference_mode():
            _ = model(dummy)
        print("✅ Model loaded & warmed up on", device)
    except Exception as e:
        # ให้เห็นใน logs ของ Render ชัด ๆ
        print("❌ Startup model load failed:", repr(e))
        model = None

def get_model():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready (startup failed)")
    return model

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "skin-maskrcnn-api", "device": device, "model_ready": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    m = get_model()

    # อ่านไฟล์แบบชัวร์ (Render/ASGI บางที file.file มี edge case)
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {repr(e)}")

    x = transform(img).unsqueeze(0).to(device)

    try:
        with torch.inference_mode():
            out = m(x)[0]
    except Exception as e:
        # ถ้าโมเดลล้ม/oom จะได้เห็น error
        raise HTTPException(status_code=500, detail=f"Inference failed: {repr(e)}")

    scores = out.get("scores", torch.empty((0,))).detach().cpu()
    labels = out.get("labels", torch.empty((0,))).detach().cpu()

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

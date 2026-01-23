from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as T
from model import load_model  # ถ้า deploy แล้วหาไม่เจอ ค่อยเปลี่ยนเป็น from skin_deploy.model import load_model

app = FastAPI(title="Skin Mask R-CNN API", version="1.0")

# -------------------------
# CONFIG
# -------------------------
CONF_THRES = 0.5   # ปรับได้ เช่น 0.3 ถ้าอยากให้เจอง่ายขึ้น
TOPK = 200         # จำกัดจำนวนผลลัพธ์สูงสุดต่อภาพ (กัน JSON ใหญ่)

# -------------------------
# CLASS NAMES (EN + TH)
# label_id ต้องตรงกับตอน train (1..7)
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

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = T.ToTensor()

model = None  # โหลดตอน startup

@app.on_event("startup")
def startup():
    global model
    model = load_model(device=device)
    model.eval()

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "skin-maskrcnn-api"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # --- read image ---
    try:
        img = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image")

    x = transform(img).unsqueeze(0).to(device)

    # --- inference ---
    with torch.inference_mode():
        out = model(x)[0]

    boxes = out.get("boxes", torch.empty((0, 4))).detach().cpu()
    scores = out.get("scores", torch.empty((0,))).detach().cpu()
    labels = out.get("labels", torch.empty((0,))).detach().cpu()

    # --- filter by confidence ---
    if len(scores) > 0:
        keep = scores >= CONF_THRES
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    # --- cap topk by score ---
    if len(scores) > TOPK:
        order = torch.argsort(scores, descending=True)[:TOPK]
        boxes = boxes[order]
        scores = scores[order]
        labels = labels[order]

    # --- build detections ---
    detections = []
    for b, s, lid in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        lid_int = int(lid)
        det = {
            "label_id": lid_int,
            "label_en": CLASS_EN.get(lid_int, f"Unknown_{lid_int}"),
            "label_th": CLASS_TH.get(lid_int, f"ไม่ทราบ_{lid_int}"),
            "score": float(s),
            "box": [float(v) for v in b],  # [x1,y1,x2,y2]
        }
        detections.append(det)

    # --- top1 ---
    top1 = detections[0] if len(detections) > 0 else None
    if top1 is None and len(scores) > 0:
        # กรณีมี output แต่ถูกกรองหมด
        top1 = {"message": f"No detections >= conf {CONF_THRES}"}

    return {
        "top1": top1
    }

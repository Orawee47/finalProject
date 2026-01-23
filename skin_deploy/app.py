from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
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

def get_model():
    global model
    if model is None:
        model = load_model(device=device)
        model.eval()
    return model

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "skin-maskrcnn-api"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    m = get_model()

    try:
        img = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image")

    x = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        out = m(x)[0]

    scores = out.get("scores", torch.empty((0,))).detach().cpu()
    labels = out.get("labels", torch.empty((0,))).detach().cpu()

    if scores.numel() == 0:
        return {"top1": None, "message": "No detections"}

    # filter by conf
    keep = scores >= CONF_THRES
    scores = scores[keep]
    labels = labels[keep]

    if scores.numel() == 0:
        return {"top1": None, "message": f"No detections >= conf {CONF_THRES}"}

    best_i = int(torch.argmax(scores).item())
    best_score = float(scores[best_i].item())
    best_label = int(labels[best_i].item())

    top1 = {
        "label_id": best_label,
        "label_en": CLASS_EN.get(best_label, f"Unknown_{best_label}"),
        "label_th": CLASS_TH.get(best_label, f"ไม่ทราบ_{best_label}"),
        "score": best_score,
    }

    return {"top1": top1}

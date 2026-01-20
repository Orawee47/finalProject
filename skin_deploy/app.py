# app.py
import io
import os
import hashlib
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import torchvision.transforms as T

from model_def import get_model, NUM_CLASSES

CLASS_NAMES_FULL = [
    "__background__", "Acne", "Herpes_Zoster", "Herpes_simplex",
    "Psoriasis", "Tinea", "Urticaria_Hives", "Vitiligo"
]

MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/best_map5095.pth")
CONF_THRES = float(os.getenv("CONF_THRES", "0.5"))
MASK_BIN_THRES = float(os.getenv("MASK_BIN_THRES", "0.5"))
MASK_ALPHA = float(os.getenv("MASK_ALPHA", "0.4"))  # ความเข้ม overlay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf = T.ToTensor()

app = FastAPI(title="Mask R-CNN Skin API")


def load_model():
    model = get_model(NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


model = load_model()


def _color_for_class(class_id: int):
    """
    สีแบบ deterministic ตาม class_id (ไม่สุ่มมั่ว)
    คืน (R,G,B)
    """
    h = hashlib.md5(str(class_id).encode()).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    # ทำให้สีไม่มืดเกิน
    r = int(80 + (r / 255) * 175)
    g = int(80 + (g / 255) * 175)
    b = int(80 + (b / 255) * 175)
    return (r, g, b)


def _overlay_masks_boxes(
    pil_img: Image.Image,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    masks_prob: np.ndarray | None,  # [N,H,W] float prob
    alpha: float = 0.4,
):
    """
    สร้าง overlay บนภาพ:
    - mask (alpha blend)
    - bbox + text
    คืน PIL Image (RGB)
    """
    img = pil_img.convert("RGB")
    base = np.array(img).astype(np.float32)  # [H,W,3]

    H, W = base.shape[0], base.shape[1]

    # --- mask overlay (เร็วและชัด) ---
    if masks_prob is not None and len(masks_prob) > 0:
        masks_bin = (masks_prob >= MASK_BIN_THRES)

        overlay = base.copy()

        for i in range(len(masks_bin)):
            m = masks_bin[i]
            if m.sum() == 0:
                continue
            c = int(labels[i])
            color = np.array(_color_for_class(c), dtype=np.float32)

            # blend เฉพาะจุดที่เป็น mask
            overlay[m] = (1 - alpha) * overlay[m] + alpha * color

        base = overlay

    out_img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

    # --- draw bbox + label ---
    draw = ImageDraw.Draw(out_img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].tolist()
        c = int(labels[i])
        s = float(scores[i])
        name = CLASS_NAMES_FULL[c] if c < len(CLASS_NAMES_FULL) else str(c)
        color = _color_for_class(c)

        # rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # label background
        text = f"{name} {s:.2f}"
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        pad = 2
        tx1, ty1 = x1, max(0, y1 - (th + pad * 2))
        tx2, ty2 = x1 + tw + pad * 2, y1
        draw.rectangle([tx1, ty1, tx2, ty2], fill=color)

        # label text
        draw.text((tx1 + pad, ty1 + pad), text, fill=(0, 0, 0), font=font)

    return out_img


@app.get("/")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "model_path": MODEL_PATH,
        "classes": CLASS_NAMES_FULL
    }


@app.post("/predict")
@torch.inference_mode()
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x = tf(img).to(device)

    out = model([x])[0]

    boxes = out["boxes"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy().astype(int)

    keep = scores >= CONF_THRES
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    preds = []
    for b, s, c in zip(boxes, scores, labels):
        preds.append({
            "class_id": int(c),
            "class_name": CLASS_NAMES_FULL[int(c)] if int(c) < len(CLASS_NAMES_FULL) else str(int(c)),
            "confidence": float(s),
            "x1": float(b[0]), "y1": float(b[1]),
            "x2": float(b[2]), "y2": float(b[3]),
        })

    return {
        "filename": file.filename,
        "n": len(preds),
        "conf_thres": CONF_THRES,
        "predictions": preds
    }


@app.post("/predict_overlay")
@torch.inference_mode()
async def predict_overlay(file: UploadFile = File(...)):
    """
    คืนรูป PNG ที่เป็น overlay (mask + box + label)
    """
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x = tf(img).to(device)

    out = model([x])[0]

    boxes = out["boxes"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy().astype(int)

    keep = scores >= CONF_THRES
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    masks_prob = None
    if "masks" in out:
        m = out["masks"].detach().cpu().numpy()  # [N,1,H,W]
        m = m[keep]
        if m.ndim == 4 and m.shape[1] == 1:
            m = m[:, 0, :, :]  # [N,H,W]
        masks_prob = m

    overlay_img = _overlay_masks_boxes(
        pil_img=img,
        boxes=boxes,
        scores=scores,
        labels=labels,
        masks_prob=masks_prob,
        alpha=MASK_ALPHA,
    )

    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

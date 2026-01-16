from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# ===== CORS (ตอนทดสอบใช้ "*" ได้ก่อน / ตอนใช้งานจริงค่อยล็อกโดเมน) =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ทดสอบก่อน
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "C:/herb_deploy/models/yolov8s(lr-4).pt"
model = YOLO(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(img)
    r = results[0]

    preds = []
    if r.boxes is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss  = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names

        for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, clss):
            preds.append({
                "class_id": int(cls_id),
                "class_name": names[int(cls_id)],
                "confidence": float(conf),
                "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            })

    return JSONResponse({"predictions": preds})

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI()

# ====== 1) CORS: อนุญาตให้เว็บ Netlify เรียก API ได้ ======
# (ถ้าคุณมีโดเมน Netlify อื่น/โดเมนจริงในอนาคต ให้เพิ่มใน list นี้ได้)
ALLOWED_ORIGINS = [
    "https://6960bb14272a930b75c8762c--skinherbcaer.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== 2) ตั้งค่าไฟล์โมเดล ======
# แนะนำให้เปลี่ยนชื่อไฟล์โมเดลให้ไม่มีวงเล็บ/ช่องว่าง เช่น YOLOv8s_lr4.pt
MODEL_PATH = os.getenv("MODEL_PATH", "models/YOLOv8s.pt")

# ====== 3) บังคับชื่อคลาสตาม data.yaml ของคุณ (ลำดับสำคัญมาก) ======
CUSTOM_NAMES = [
    "Alovera",              # id 0
    "cucumber",             # id 1
    "Galanga",              # id 2
    "Garlic",               # id 3
    "horapa",               # id 4
    "Houttuynia_cordata",   # id 5
    "Ivy_Gourd",            # id 6
    "khaprao",              # id 7
    "Mangosteen_Peel",      # id 8
    "pluleaf",              # id 9
    "Snake_Plant",          # id 10
    "Turmeric",             # id 11
]

model = None


@app.on_event("startup")
def load_model():
    global model
    print(f"🚀 Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded.")
    try:
        print("ℹ️ model.names (from pt):", model.names)
    except Exception as e:
        print("⚠️ Could not read model.names:", e)
    print("✅ CUSTOM_NAMES (forced):", CUSTOM_NAMES)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    รับรูปภาพ -> ส่งผลลัพธ์เป็น JSON:
    predictions: [{class_id, class_name, confidence, box_xyxy}]
    """
    try:
        if model is None:
            return JSONResponse(status_code=503, content={"error": "Model not loaded yet"})

        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ปรับค่าตรงนี้ได้ตามต้องการ
        # imgsz ยิ่งเล็กยิ่งเร็ว (แต่ความแม่นอาจลด)
        results = model(img, imgsz=640, conf=0.25, device="cpu")
        r = results[0]

        preds = []
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls)
                class_name = CUSTOM_NAMES[cls_id] if 0 <= cls_id < len(CUSTOM_NAMES) else "unknown"

                preds.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": float(box.conf),
                    "box_xyxy": box.xyxy.tolist()[0],  # [x1,y1,x2,y2]
                })

        return {"predictions": preds}

    except Exception as e:
        # ส่ง error ออกไปให้เห็นชัด (และ Render Logs จะมีด้วย)
        print("❌ /predict ERROR:", repr(e))
        return JSONResponse(status_code=500, content={"error": str(e)})


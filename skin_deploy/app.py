from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as T
from model import load_model  # ถ้า deploy แล้วหาไม่เจอ ค่อยเปลี่ยนเป็น from skin_deploy.model import load_model

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = T.ToTensor()

model = None  # จะโหลดตอน startup

@app.on_event("startup")
def startup():
    global model
    model = load_model(device=device)
    model.eval()
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "skin-maskrcnn-api"
    }
    
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        img = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)[0]

    return {
        "boxes": output["boxes"].detach().cpu().tolist(),
        "scores": output["scores"].detach().cpu().tolist(),
        "labels": output["labels"].detach().cpu().tolist()
    }

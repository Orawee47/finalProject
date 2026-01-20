from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.transforms as T
from model import load_model

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(device=device)
transform = T.ToTensor()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)[0]

    return {
        "boxes": output["boxes"].cpu().tolist(),
        "scores": output["scores"].cpu().tolist(),
        "labels": output["labels"].cpu().tolist()
    }

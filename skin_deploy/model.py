import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import os
import urllib.request

MODEL_URL = os.environ.get("MODEL_URL")
MODEL_PATH = "https://github.com/Orawee47/finalProject/releases/download/v1.0-maskrcnn/best_map5095.pth"

def load_model(num_classes=8, device="cpu"):
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

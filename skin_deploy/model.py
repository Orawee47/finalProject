import os
import torch
import urllib.request
from torchvision.models.detection import maskrcnn_resnet50_fpn

MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/Orawee47/finalProject/releases/download/v1.0-maskrcnn/best_map5095.pth"
)

# ✅ ใช้ path ภายในโปรเจค (Render / Docker ปลอดภัย)
MODEL_DIR = "models"
MODEL_FILENAME = "best_map5095.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


def load_model(num_classes=8, device="cpu"):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")
        print(f"From: {MODEL_URL}")
        print(f"To:   {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model

import os
import urllib.request
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn

# ✅ ถ้ามี ENV ก็ใช้ ENV ไม่มีก็ใช้ค่า default
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/Orawee47/finalProject/releases/download/v1.0-maskrcnn/best_map5095.pth"
)

# ✅ PATH ต้องเป็น path ในเครื่อง ไม่ใช่ URL
MODEL_DIR = "C:\skin_train_10_01_69\results_maskrcnn\run_baseline_lr005\checkpoints"
MODEL_FILENAME = "best_map5095.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

def load_model(num_classes=8, device="cpu"):
    # ✅ สร้างโฟลเดอร์ไว้ก่อน กัน FileNotFoundError
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ✅ ถ้ายังไม่มีไฟล์ค่อยโหลด
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Model downloaded:", MODEL_PATH)

    # ✅ สร้างโมเดลให้ตรงกับตอนเทรน (num_classes ต้องตรง)
    model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)

    # ✅ โหลด state_dict
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)

    model.to(device).eval()
    return model

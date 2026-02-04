# model.py
import os
from pathlib import Path
import requests
import torch

# ✅ แนะนำ: อย่า import torchvision ไวเกินจำเป็น (ค่อย import ใน build_model)
# จะช่วยลด peak RAM ตอนเริ่ม

MODEL_URL = os.environ.get("MODEL_URL", "https://github.com/Orawee47/finalProject/releases/download/v1.1-maskrcnn/best.pth")  # direct download ของ release asset
CACHE_DIR = Path(os.environ.get("MODEL_DIR", "/tmp/models"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = CACHE_DIR / "best.pth"


def download_if_needed():
    # ถ้ามีไฟล์แล้วและดูไม่เล็กผิดปกติ ก็ใช้เลย
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 10_000_000:
        return str(MODEL_PATH)

    if not MODEL_URL:
        raise RuntimeError("MODEL_URL is empty. Set MODEL_URL in Render env variables.")

    tmp_path = MODEL_PATH.with_suffix(".tmp")

    # ✅ stream download ลงดิสก์ ไม่โหลดทั้งไฟล์เข้า RAM
    with requests.get(MODEL_URL, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB
                if chunk:
                    f.write(chunk)

    tmp_path.replace(MODEL_PATH)
    return str(MODEL_PATH)


def build_model(num_classes: int):
    """
    ⚠️ ต้องให้ "ตรงกับตอนเทรน"
    ถ้าตอนเทรนใช้ maskrcnn_resnet50_fpn(weights=None, num_classes=NUM_CLASSES)
    ก็ใช้แบบนี้ได้เลย
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn

    model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    return model


def load_model(device="cpu", num_classes=8):
    # ลด peak RAM/CPU thread
    torch.set_num_threads(1)

    ckpt_path = download_if_needed()

    model = build_model(num_classes=num_classes)

    # ✅ สำคัญ: map_location=cpu เพื่อลดการก็อปปี้/ย้าย device ตอนโหลด
    # ✅ weights_only=True (ถ้า torch รองรับ) ลดความเสี่ยงโหลด object อื่น ๆ
    try:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        # เผื่อ torch รุ่นเก่าไม่รองรับ weights_only
        state_dict = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

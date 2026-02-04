import os
from pathlib import Path
import requests
import torch

# ใส่ URL direct download ของ release asset (.pth)
MODEL_URL = os.environ.get("MODEL_URL", "")
CACHE_DIR = Path(os.environ.get("MODEL_DIR", "/tmp/models"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = CACHE_DIR / "best.pth"


def download_if_needed():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 10_000_000:
        return str(MODEL_PATH)

    if not MODEL_URL:
        raise RuntimeError("MODEL_URL is empty. Set MODEL_URL in Render env variables.")

    tmp_path = MODEL_PATH.with_suffix(".tmp")

    # stream download ลงดิสก์ ไม่โหลดทั้งไฟล์เข้า RAM
    with requests.get(MODEL_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB
                if chunk:
                    f.write(chunk)

    tmp_path.replace(MODEL_PATH)
    return str(MODEL_PATH)


def build_model(num_classes: int):
    """
    ให้ตรงกับตอนเทรนของคุณ:
    - maskrcnn_resnet50_fpn(weights=DEFAULT)
    - เปลี่ยน box_predictor และ mask_predictor
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    # ตอนเทรนคุณใช้ weights=DEFAULT → แต่ตอน deploy แนะนำใช้ weights=None กันโหลดซ้ำ
    model = maskrcnn_resnet50_fpn(weights=None)

    # replace box head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # replace mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def load_model(device="cpu", num_classes=8):
    # ลด peak RAM เวลาเริ่ม
    torch.set_num_threads(1)

    ckpt_path = download_if_needed()
    model = build_model(num_classes=num_classes)

    # สำคัญ: map_location=cpu + weights_only=True (ลด peak)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

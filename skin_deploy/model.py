# model.py
import os
from pathlib import Path
import gc
import requests
import torch

MODEL_URL = os.environ.get("MODEL_URL", "https://github.com/Orawee47/finalProject/releases/download/v1.1-maskrcnn/best.pth")
CACHE_DIR = Path(os.environ.get("MODEL_DIR", "/tmp/models"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = CACHE_DIR / "best.pth"

def download_if_needed():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 10_000_000:
        return str(MODEL_PATH)

    if not MODEL_URL:
        raise RuntimeError("MODEL_URL is empty. Set MODEL_URL in env variables.")

    tmp_path = MODEL_PATH.with_suffix(".tmp")

    with requests.get(MODEL_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    tmp_path.replace(MODEL_PATH)
    return str(MODEL_PATH)

def build_model(num_classes: int):
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    # สำคัญ: weights=None, weights_backbone=None เพื่อไม่โหลด pretrained ซ้ำ (ลด RAM)
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    # replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden, num_classes)

    return model

def load_model(device="cpu", num_classes=8):
    torch.set_num_threads(1)  # ลด RAM/CPU spike
    ckpt_path = download_if_needed()

    model = build_model(num_classes=num_classes)

    # ลด peak memory: โหลดขึ้น CPU ก่อนเสมอ
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    del state_dict
    gc.collect()

    model.to(device)
    model.eval()
    return model

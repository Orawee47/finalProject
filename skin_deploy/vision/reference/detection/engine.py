# vision/references/detection/engine.py

import math
import sys
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

import utils
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator


def _is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def _is_main_process():
    if not _is_dist_avail_and_initialized():
        return True
    return dist.get_rank() == 0


def reduce_dict(input_dict, average=True):
    if not _is_dist_avail_and_initialized():
        return input_dict

    with torch.inference_mode():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= dist.get_world_size()
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = f"Epoch: [{epoch}]"
    is_main = _is_main_process()

    pbar = tqdm(
        total=len(data_loader),
        desc=header,
        leave=False,              # ✅ ไม่ทิ้งบรรทัดค้าง
        dynamic_ncols=True,       # ✅ กัน bar ล้น
        mininterval=0.2,          # ✅ ลด refresh ถี่
        disable=not is_main       # ✅ กันหลาย process
    )

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # ✅ แก้ warning autocast (PyTorch ใหม่)
        use_amp = scaler is not None
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # ✅ ตั้ง postfix ไม่ถี่เกิน
        if i % max(1, print_freq) == 0:
            pbar.set_postfix_str(
                f"loss={loss_value:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        pbar.update(1)

    pbar.close()
    return metric_logger


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    model.eval()
    coco = get_coco_api_from_dataset(data_loader.dataset)

    # ✅ ไม่เรียก utils._get_iou_types เพื่อกัน utils ชนกัน
    iou_types = ["bbox"]
    if hasattr(model, "roi_heads") and getattr(model.roi_heads, "mask_predictor", None) is not None:
        iou_types.append("segm")

    coco_evaluator = CocoEvaluator(coco, iou_types)

    is_main = _is_main_process()
    pbar = tqdm(
        total=len(data_loader),
        desc="Eval",
        leave=False,              # ✅ ไม่ทิ้งบรรทัดค้าง
        dynamic_ncols=True,
        mininterval=0.2,
        disable=not is_main
    )

    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        outputs = model(images)
        outputs = [{k: v.to("cpu") for k, v in o.items()} for o in outputs]

        # targets ต้องมี image_id เป็น scalar tensor
        res = {int(t["image_id"].item()): out for t, out in zip(targets, outputs)}
        coco_evaluator.update(res)

        pbar.update(1)

    pbar.close()

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    torch.set_num_threads(n_threads)
    return coco_evaluator

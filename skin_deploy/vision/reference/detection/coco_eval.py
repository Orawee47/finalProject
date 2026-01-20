import copy
import io
from contextlib import redirect_stdout

import numpy as np
import pycocotools.mask as mask_util
import torch
import utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(
                f"This constructor expects iou_types of type list or tuple, instead got {type(iou_types)}"
            )

        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        # ✅ FIX: กัน COCO GT json ที่ไม่มี "info" / "licenses"
        # pycocotools.COCO.loadRes จะพยายาม copy self.dataset['info']
        if not hasattr(self.coco_gt, "dataset") or self.coco_gt.dataset is None:
            self.coco_gt.dataset = {}
        if "info" not in self.coco_gt.dataset:
            self.coco_gt.dataset["info"] = {}
        if "licenses" not in self.coco_gt.dataset:
            self.coco_gt.dataset["licenses"] = []

        self.iou_types = iou_types
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type) for iou_type in iou_types}

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # COCO.loadRes() จะพังถ้า results ว่าง -> ใช้ COCO() ได้
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()

            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)

            img_ids_, eval_imgs = evaluate(coco_eval)
            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = convert_to_xywh(prediction["boxes"]).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": int(original_id),
                        "category_id": int(labels[k]),
                        "bbox": box,
                        "score": float(scores[k]),
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            masks = prediction["masks"]  # torch tensor [N,1,H,W] or [N,H,W]
            if isinstance(masks, torch.Tensor):
                masks = masks.detach().cpu().numpy()

            # ให้เป็น [N,H,W]
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0, :, :]

            # binary
            masks = masks > 0.5

            # ✅ FIX: encode mask ต้องเป็น Fortran order [H,W]
            rles = [mask_util.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": int(original_id),
                        "category_id": int(labels[k]),
                        "segmentation": rles[k],
                        "score": float(scores[k]),
                    }
                    for k in range(len(rles))
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = convert_to_xywh(prediction["boxes"]).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": int(original_id),
                        "category_id": int(labels[k]),
                        "keypoints": keypoints[k],
                        "score": float(scores[k]),
                    }
                    for k in range(len(keypoints))
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(coco_eval):
    with redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
    return coco_eval.params.imgIds, np.asarray(coco_eval.evalImgs).reshape(
        -1, len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)
    )

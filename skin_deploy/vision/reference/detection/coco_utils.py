# vision/references/detection/coco_utils.py
# Fixed version for custom Mask R-CNN training/evaluation with COCO API
# Works with datasets that return (image, target) where target has:
#   boxes (FloatTensor[N,4]), labels (IntTensor[N]), masks (UInt8Tensor[N,H,W]), image_id (scalar tensor or int)
#
# Key fixes:
# - ensure image_id is int in COCO dataset
# - ensure target["image_id"] is scalar tensor for engine/evaluate
# - safe mask encoding for pycocotools (uint8 + fortran contiguous)

import copy
import numpy as np
import torch
import torch.utils.data
import torchvision

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO


def _to_int_image_id(image_id):
    """Convert image_id (int / scalar tensor / tensor shape [1]) -> python int."""
    if torch.is_tensor(image_id):
        if image_id.numel() == 1:
            return int(image_id.reshape(-1)[0].item())
        # fall back: take first element
        return int(image_id.reshape(-1)[0].item())
    return int(image_id)


def _to_scalar_tensor_image_id(image_id):
    """Convert image_id (int / tensor [1] / scalar) -> scalar int64 tensor."""
    if torch.is_tensor(image_id):
        image_id = image_id.to(dtype=torch.int64)
        if image_id.ndim == 0:
            return image_id
        return image_id.reshape(-1)[0]
    return torch.tensor(int(image_id), dtype=torch.int64)


def convert_coco_poly_to_mask(segmentations, height, width):
    """Convert COCO polygon to binary masks (UInt8Tensor[N,H,W])."""
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        rle = coco_mask.merge(rles)
        m = coco_mask.decode(rle)
        if m.ndim < 3:
            m = m[..., None]
        m = torch.as_tensor(m, dtype=torch.uint8)
        m = m.any(dim=2)  # [H,W]
        masks.append(m)
    if len(masks) == 0:
        return torch.zeros((0, height, width), dtype=torch.uint8)
    return torch.stack(masks, dim=0)


class ConvertCocoPolysToMask:
    """
    Used for torchvision.datasets.CocoDetection style dataset.
    Keeps image_id as scalar tensor and returns target with boxes/labels/masks.
    """

    def __call__(self, image, target):
        # target: dict(image_id=..., annotations=[...])
        w, h = image.size

        image_id = _to_scalar_tensor_image_id(target["image_id"])
        anno = target["annotations"]

        # filter crowd
        anno = [obj for obj in anno if obj.get("iscrowd", 0) == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # xywh -> xyxy
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.as_tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # keep only valid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        area = torch.tensor([obj.get("area", 0.0) for obj in anno], dtype=torch.float32)
        iscrowd = torch.zeros((len(anno),), dtype=torch.int64)
        if len(area) == len(keep):
            area = area[keep]

        out_target = {}
        out_target["boxes"] = boxes
        out_target["labels"] = classes
        out_target["masks"] = masks
        out_target["image_id"] = image_id
        out_target["area"] = area
        out_target["iscrowd"] = iscrowd

        if keypoints is not None:
            out_target["keypoints"] = keypoints

        return image, out_target


class CocoDetection(torchvision.datasets.CocoDetection):
    """
    Wrapper for torchvision.datasets.CocoDetection to output Mask R-CNN targets.
    """

    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask()

    def __getitem__(self, idx):
        img, ann = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=ann)

        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img = self._transforms(img)

        return img, target


def get_coco_api_from_dataset(dataset):
    """
    Returns COCO API object for a dataset.
    If dataset is CocoDetection wrapper, just return its COCO object.
    Otherwise, convert via convert_to_coco_api().
    """
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            return dataset.coco
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        else:
            break
    return convert_to_coco_api(dataset)


def convert_to_coco_api(ds):
    """
    Convert a dataset that yields (image, target) into a COCO-style API object.
    This is used by torchvision reference evaluate().
    """
    coco_ds = COCO()
    coco_ds.dataset = {"images": [], "annotations": [], "categories": []}

    ann_id = 1
    categories = set()

    for img_idx in range(len(ds)):
        _, targets = ds[img_idx]

        # -------- image_id must be int for COCO dataset --------
        image_id = _to_int_image_id(targets["image_id"])

        # infer H,W
        if "masks" in targets and torch.is_tensor(targets["masks"]) and targets["masks"].numel() > 0:
            h, w = targets["masks"].shape[-2], targets["masks"].shape[-1]
        else:
            # fallback: if boxes exist but no masks
            # (not typical for Mask R-CNN training, but keep safe)
            h, w = 0, 0

        img_dict = {"id": image_id}
        if h and w:
            img_dict.update({"height": int(h), "width": int(w)})
        coco_ds.dataset["images"].append(img_dict)

        # targets fields
        boxes = targets.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
        labels = targets.get("labels", torch.zeros((0,), dtype=torch.int64))
        masks = targets.get("masks", torch.zeros((0, h, w), dtype=torch.uint8))
        area = targets.get("area", None)
        iscrowd = targets.get("iscrowd", torch.zeros((len(labels),), dtype=torch.int64))

        if torch.is_tensor(labels):
            labels_list = labels.tolist()
        else:
            labels_list = list(labels)

        categories.update(labels_list)

        # make sure shapes
        if torch.is_tensor(boxes):
            boxes_np = boxes.detach().cpu().numpy()
        else:
            boxes_np = np.asarray(boxes)

        # xyxy -> xywh
        if boxes_np.size > 0:
            boxes_xywh = boxes_np.copy()
            boxes_xywh[:, 2] = boxes_xywh[:, 2] - boxes_xywh[:, 0]
            boxes_xywh[:, 3] = boxes_xywh[:, 3] - boxes_xywh[:, 1]
        else:
            boxes_xywh = boxes_np.reshape(0, 4)

        # area fallback
        if area is None:
            if boxes_xywh.size > 0:
                area_vals = (boxes_xywh[:, 2] * boxes_xywh[:, 3]).astype(np.float32)
            else:
                area_vals = np.zeros((0,), dtype=np.float32)
        else:
            if torch.is_tensor(area):
                area_vals = area.detach().cpu().numpy().astype(np.float32)
            else:
                area_vals = np.asarray(area, dtype=np.float32)

        # iscrowd
        if torch.is_tensor(iscrowd):
            iscrowd_vals = iscrowd.detach().cpu().numpy().astype(np.int64)
        else:
            iscrowd_vals = np.asarray(iscrowd, dtype=np.int64)

        # masks encode (optional)
        masks_np = None
        if torch.is_tensor(masks):
            masks_np = masks.detach().cpu().numpy().astype(np.uint8)
        elif masks is not None:
            masks_np = np.asarray(masks, dtype=np.uint8)

        num_objs = len(labels_list)

        for i in range(num_objs):
            label_i = int(labels_list[i])
            if label_i == 0:
                # skip background if ever present
                continue

            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": label_i,
                "bbox": boxes_xywh[i].tolist() if boxes_xywh.size else [0, 0, 0, 0],
                "area": float(area_vals[i]) if len(area_vals) > i else 0.0,
                "iscrowd": int(iscrowd_vals[i]) if len(iscrowd_vals) > i else 0,
            }

            if masks_np is not None and masks_np.shape[0] > i:
                m = masks_np[i]
                # pycocotools wants Fortran contiguous
                rle = coco_mask.encode(np.asfortranarray(m))
                # encode returns bytes, convert to str for json compatibility
                rle["counts"] = rle["counts"].decode("utf-8")
                ann["segmentation"] = rle

            coco_ds.dataset["annotations"].append(ann)
            ann_id += 1

    coco_ds.dataset["categories"] = [{"id": int(i)} for i in sorted(categories)]
    coco_ds.createIndex()
    return coco_ds

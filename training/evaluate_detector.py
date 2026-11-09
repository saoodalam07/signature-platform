"""Evaluate a trained RF-DETR checkpoint on the COCO test split.

Reports mAP50, mAP50-95 (pycocotools, BSD) plus precision / recall / F1 at the app threshold,
so the numbers are directly comparable with the old YOLO26s results.

    python -m training.evaluate_detector --weights runs/.../checkpoint_best_total.pth --data signature_dataset_coco
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def evaluate(weights, data_dir, split="test", threshold=0.40, limit=None, resolution=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from rfdetr import RFDETRLarge

    ann_path = os.path.join(data_dir, split, "_annotations.coco.json")
    gt = COCO(ann_path)
    sig_cats = sorted({a["category_id"] for a in gt.dataset["annotations"]})
    cat_id = sig_cats[0]   # single class: every prediction is a signature

    kw = {"resolution": resolution} if resolution else {}
    model = RFDETRLarge(pretrain_weights=weights, trust_checkpoint=True, **kw)   # our own checkpoint
    img_ids = gt.getImgIds()[:limit] if limit else gt.getImgIds()
    preds, tp, fp, fn, times = [], 0, 0, 0, []

    for i, img_id in enumerate(img_ids, 1):
        info = gt.loadImgs(img_id)[0]
        img = Image.open(os.path.join(data_dir, split, info["file_name"])).convert("RGB")
        t0 = time.perf_counter()
        d = model.predict(img, threshold=0.01)
        times.append((time.perf_counter() - t0) * 1000)
        boxes = [list(map(float, b)) for b in d.xyxy]
        confs = [float(c) for c in d.confidence]
        for b, c in zip(boxes, confs):
            preds.append({"image_id": img_id, "category_id": cat_id,
                          "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": c})
        # precision / recall at the operating threshold (greedy IoU>=0.5 matching)
        gts = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
               for a in gt.loadAnns(gt.getAnnIds(imgIds=img_id))]
        kept = sorted([(c, b) for b, c in zip(boxes, confs) if c >= threshold], reverse=True)
        used = set()
        for c, b in kept:
            best, j = 0.0, -1
            for k, g in enumerate(gts):
                if k not in used and iou(b, g) > best:
                    best, j = iou(b, g), k
            if best >= 0.5:
                tp += 1
                used.add(j)
            else:
                fp += 1
        fn += len(gts) - len(used)
        if i % 200 == 0:
            print(f"  {i}/{len(img_ids)}")

    result = {"split": split, "images": len(img_ids), "instances": tp + fn,
              "map50": 0.0, "map50_95": 0.0}
    if preds:
        dt = gt.loadRes(preds)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = img_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
        result["map50_95"], result["map50"] = float(ev.stats[0]), float(ev.stats[1])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    result.update({
        "threshold": threshold, "precision": p, "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "ms_per_image": float(np.median(times[5:] or times)),
    })
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="signature_dataset_coco")
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--resolution", type=int, default=None, help="must match training (e.g. 640)")
    a = ap.parse_args()
    print(json.dumps(evaluate(a.weights, a.data, a.split, a.threshold, resolution=a.resolution), indent=2))

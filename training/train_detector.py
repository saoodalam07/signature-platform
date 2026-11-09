"""Train RF-DETR Large on the COCO signature dataset, track in MLflow, register the model.

    python -m training.train_detector --data signature_dataset_coco
    python -m training.train_detector --data signature_dataset_coco --resume

Every run:
  1. trains RF-DETR Large (Apache-2.0) with per-epoch metrics logged to MLflow,
  2. evaluates the best checkpoint on the held-out test split,
  3. registers it as a new version of "signature-detector" in the MLflow Model Registry,
  4. promotes it to @production only if it beats the current production model.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sigplatform.config import settings            # noqa: E402
from sigplatform.ml import registry                # noqa: E402
from training.evaluate_detector import evaluate    # noqa: E402


def check_dataset(data):
    for split in ("train", "valid", "test"):
        p = os.path.join(data, split, "_annotations.coco.json")
        if not os.path.exists(p):
            sys.exit(f"Missing {p}")
        d = json.load(open(p, encoding="utf-8"))
        cats = [c["name"] for c in d.get("categories", [])]
        print(f"  {split:5s}: {len(d['images']):6d} images  {len(d['annotations']):6d} boxes  classes={cats}")


def make_subset(data, train_limit=None, valid_limit=None, seed=0):
    """Random subset of train/valid for fast runs. Images are hard-linked (no extra disk space).
    The test split is NOT reduced: final scores are always on the full test set."""
    import random
    import shutil
    if not train_limit and not valid_limit:
        return data
    out = os.path.join(os.path.dirname(os.path.abspath(data)),
                       f"{os.path.basename(os.path.abspath(data))}_subset_t{train_limit or 'all'}_v{valid_limit or 'all'}")
    for split, limit in (("train", train_limit), ("valid", valid_limit), ("test", None)):
        src_dir, dst_dir = os.path.join(data, split), os.path.join(out, split)
        os.makedirs(dst_dir, exist_ok=True)
        d = json.load(open(os.path.join(src_dir, "_annotations.coco.json"), encoding="utf-8"))
        imgs = d["images"]
        if limit and limit < len(imgs):
            imgs = random.Random(seed).sample(imgs, limit)
        keep = {im["id"] for im in imgs}
        sub = dict(d, images=imgs, annotations=[a for a in d["annotations"] if a["image_id"] in keep])
        json.dump(sub, open(os.path.join(dst_dir, "_annotations.coco.json"), "w", encoding="utf-8"))
        for im in imgs:
            s_, t_ = os.path.join(src_dir, im["file_name"]), os.path.join(dst_dir, im["file_name"])
            if not os.path.exists(t_):
                os.makedirs(os.path.dirname(t_), exist_ok=True)
                try:
                    os.link(s_, t_)
                except OSError:
                    shutil.copy2(s_, t_)
        print(f"  subset {split}: {len(imgs)} images")
    return out


def train_and_register(data, out_dir, epochs, batch, resume=False, run_name=None, dataset_rev=None,
                       resolution=None, patience=None, multiscale=True, workers=2,
                       train_limit=None, valid_limit=None):
    resolution = resolution or settings.train_resolution
    patience = patience or settings.train_patience
    import mlflow
    import torch
    from rfdetr import RFDETRLarge   # Large = Apache-2.0. Never XL/2XL (PML licence).

    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found. Install the CUDA build of PyTorch.")
    print("GPU:", torch.cuda.get_device_name(0))
    check_dataset(data)
    full_data = data
    data = make_subset(data, train_limit, valid_limit)

    os.environ["MLFLOW_TRACKING_URI"] = settings.mlflow_uri
    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.experiment)
    run_name = run_name or os.path.basename(out_dir.rstrip("/\\"))

    last = os.path.join(out_dir, "last.ckpt")
    model = RFDETRLarge()
    model.train(
        dataset_dir=data,
        dataset_file="roboflow",
        output_dir=out_dir,
        epochs=epochs,
        batch_size=batch,
        grad_accum_steps=max(1, 16 // batch),     # effective batch 16
        early_stopping=True,
        early_stopping_patience=patience,
        resolution=resolution,                    # must be divisible by 32
        multi_scale="per-batch" if multiscale else "off",
        num_workers=workers,
        run_test=False,                           # we evaluate on the FULL test set below
        tensorboard=True,
        mlflow=True,                              # per-epoch curves (RF-DETR's built-in logger)
        project=settings.experiment,
        run=run_name,
        resume=last if resume and os.path.exists(last) else None,
    )

    best = os.path.join(out_dir, "checkpoint_best_total.pth")
    if not os.path.exists(best):
        sys.exit(f"Best checkpoint not found in {out_dir}")

    print("\nEvaluating best checkpoint on the test split...")
    metrics = evaluate(best, full_data, "test", settings.detect_threshold, resolution=resolution)
    print(json.dumps(metrics, indent=2))

    # attach final metrics + model to the run RF-DETR created (same run name)
    runs = mlflow.search_runs(experiment_names=[settings.experiment],
                              filter_string=f"tags.mlflow.runName = '{run_name}'",
                              order_by=["start_time DESC"], max_results=1)
    run_id = runs.iloc[0].run_id if len(runs) else None
    with mlflow.start_run(run_id=run_id, run_name=run_name):
        mlflow.log_params({"model": "rfdetr-large", "licence": "Apache-2.0", "epochs": epochs,
                           "batch": batch, "train_limit": train_limit or "all", "valid_limit": valid_limit or "all", "resolution": resolution, "patience": patience, "multiscale": multiscale, "dataset": data, "dataset_revision": dataset_rev or "local"})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
        version = registry.log_and_register(
            settings.detector_model_name, {"weights": best},
            {"test_map50": metrics["map50"], "test_map50_95": metrics["map50_95"],
             "dataset_revision": dataset_rev or "local", "arch": "rfdetr-large", "resolution": resolution})

    promoted = maybe_promote(settings.detector_model_name, version, metrics["map50"])
    return {"version": version, "promoted": promoted, **metrics}


def maybe_promote(name, version, new_map50):
    cur, tags = registry.production_version(name)
    cur_map = float(tags.get("test_map50", -1)) if cur else -1
    if cur is None or new_map50 >= cur_map + settings.promote_min_gain:
        registry.promote(name, version)
        print(f"PROMOTED v{version} (mAP50 {new_map50:.4f} vs production {cur_map:.4f})")
        return True
    print(f"Kept production v{cur}: new mAP50 {new_map50:.4f} < {cur_map:.4f}")
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="signature_dataset_coco")
    ap.add_argument("--out", default=os.path.join("runs_rfdetr", "signature_rfdetr_large"))
    ap.add_argument("--epochs", type=int, default=settings.train_epochs)
    ap.add_argument("--batch", type=int, default=settings.train_batch, help="use 2 on CUDA out-of-memory")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resolution", type=int, default=settings.train_resolution, help="704 = RF-DETR default, 640 faster, 576 fastest")
    ap.add_argument("--patience", type=int, default=settings.train_patience, help="stop after N epochs without improvement")
    ap.add_argument("--no-multiscale", action="store_true", help="train at one fixed size (faster)")
    ap.add_argument("--workers", type=int, default=4, help="data-loading processes")
    ap.add_argument("--train-limit", type=int, default=None, help="use only N random training images (fast run)")
    ap.add_argument("--valid-limit", type=int, default=None, help="use only N validation images per epoch")
    a = ap.parse_args()
    if a.resolution % 32:
        sys.exit("--resolution must be divisible by 32 (e.g. 576, 608, 640, 704)")
    print(json.dumps(train_and_register(a.data, a.out, a.epochs, a.batch, a.resume, resolution=a.resolution,
                                        patience=a.patience, multiscale=not a.no_multiscale,
                                        workers=a.workers, train_limit=a.train_limit,
                                        valid_limit=a.valid_limit), indent=2))

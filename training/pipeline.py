"""End-to-end training pipeline:

  Hugging Face dataset updated -> download that exact revision -> validate -> train RF-DETR Large
  -> evaluate on test -> register in MLflow -> promote to @production if better
  -> inference workers hot-reload the new model (they poll the @production alias).

Run by hand:
    python -m training.pipeline --repo sevenminds/signature-dataset
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sigplatform.config import settings                       # noqa: E402
from training.train_detector import train_and_register        # noqa: E402


def download_dataset(repo, revision=None):
    from huggingface_hub import snapshot_download
    # one folder per repo (not per revision): a new revision only downloads the files that changed
    local = os.path.join(settings.work_dir, "datasets", repo.replace("/", "__"))
    snapshot_download(repo_id=repo, repo_type="dataset", revision=revision, local_dir=local,
                      token=settings.hf_token)
    # accept either the repo root or a single sub-folder holding train/valid/test
    for cand in [local] + [os.path.join(local, d) for d in os.listdir(local)]:
        if all(os.path.exists(os.path.join(cand, s, "_annotations.coco.json")) for s in ("train", "valid", "test")):
            return cand
    raise RuntimeError(f"{repo}@{revision or 'main'}: expected train/ valid/ test/ folders with _annotations.coco.json")


def run(repo, revision=None):
    data = download_dataset(repo, revision)
    rev = (revision or "main")[:12]
    out = os.path.join(settings.work_dir, "runs", f"rfdetr_large_{rev}")
    result = train_and_register(data, out, settings.train_epochs, settings.train_batch,
                                run_name=f"auto-{rev}", dataset_rev=f"{repo}@{revision or 'main'}")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=settings.hf_dataset_repo, required=settings.hf_dataset_repo is None)
    ap.add_argument("--revision", default=None)
    a = ap.parse_args()
    run(a.repo, a.revision)

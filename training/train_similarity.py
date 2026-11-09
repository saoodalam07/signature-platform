"""Signature similarity: DINOv2 ViT-B (frozen, Apache-2.0) + a small projection head trained with triplet loss.

Dataset layout (one folder per person; "forged" is optional but strongly recommended):

    signature_verification/
        train/<person_id>/genuine/*.png    train/<person_id>/forged/*.png
        val/<person_id>/genuine/*.png      val/<person_id>/forged/*.png

Public sources with this structure: CEDAR, BHSig260, SigComp (check each licence before production use).
People in val must not appear in train.

    python -m training.train_similarity --data signature_verification

Logs to MLflow: zero-shot baseline vs fine-tuned EER / ROC-AUC, and the best match threshold.
Registers "signature-similarity" and promotes it if its EER beats production.
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sigplatform.config import settings     # noqa: E402
from sigplatform.ml import registry         # noqa: E402

IMG = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp")


def load_split(root):
    people = {}
    for pdir in sorted(glob.glob(os.path.join(root, "*"))):
        g = sorted(sum([glob.glob(os.path.join(pdir, "genuine", e)) for e in IMG], []))
        f = sorted(sum([glob.glob(os.path.join(pdir, "forged", e)) for e in IMG], []))
        if len(g) >= 2:
            people[os.path.basename(pdir)] = {"genuine": g, "forged": f}
    return people


def features(embedder, people):
    from PIL import Image
    paths = [p for v in people.values() for k in ("genuine", "forged") for p in v[k]]
    feats = embedder.raw_features([Image.open(p).convert("RGB") for p in paths])
    return dict(zip(paths, feats))


def eval_pairs(people, emb):
    """Genuine-genuine pairs (label 1) vs genuine-forged / other-person pairs (label 0)."""
    from sklearn.metrics import roc_auc_score, roc_curve
    rng = random.Random(0)
    ids, scores, labels = list(people), [], []
    for pid, v in people.items():
        g = v["genuine"]
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                scores.append(float(emb[g[i]] @ emb[g[j]])); labels.append(1)
            for f in v["forged"]:
                scores.append(float(emb[g[i]] @ emb[f])); labels.append(0)
            other = rng.choice([o for o in ids if o != pid])
            scores.append(float(emb[g[i]] @ emb[rng.choice(people[other]["genuine"])])); labels.append(0)
    scores, labels = np.array(scores), np.array(labels)
    fpr, tpr, thr = roc_curve(labels, scores)
    k = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    return {"roc_auc": float(roc_auc_score(labels, scores)), "eer": float((fpr[k] + 1 - tpr[k]) / 2),
            "threshold": float(thr[k]), "pairs": int(len(labels))}


def normalised(feat_map, head=None):
    import torch
    out = {}
    with torch.no_grad():
        for p, f in feat_map.items():
            x = f.unsqueeze(0)
            if head is not None:
                x = head(x)
            out[p] = torch.nn.functional.normalize(x, dim=1)[0].cpu().numpy()
    return out


def train_head(train_people, feats, epochs=30, out_dim=256, margin=0.3, device="cuda"):
    import torch
    from sigplatform.ml.similarity_head import ProjectionHead
    in_dim = next(iter(feats.values())).shape[0]
    head = ProjectionHead(in_dim, out_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=lambda a, b: 1 - torch.nn.functional.cosine_similarity(a, b), margin=margin)
    ids, rng = list(train_people), random.Random(0)
    for ep in range(epochs):
        trip = []
        for pid, v in train_people.items():
            for a in v["genuine"]:
                p = rng.choice([x for x in v["genuine"] if x != a])
                if v["forged"] and rng.random() < 0.7:          # skilled forgery = hardest negative
                    n = rng.choice(v["forged"])
                else:
                    n = rng.choice(train_people[rng.choice([o for o in ids if o != pid])]["genuine"])
                trip.append((a, p, n))
        rng.shuffle(trip)
        head.train(); total = 0.0
        for i in range(0, len(trip), 128):
            b = trip[i:i + 128]
            A, P, N = (torch.stack([feats[t[k]] for t in b]).to(device) for k in range(3))
            loss = loss_fn(head(A), head(P), head(N))
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item() * len(b)
        yield ep, total / max(1, len(trip)), head.eval(), in_dim, out_dim


def main(data, epochs, out_dir):
    import mlflow
    from sigplatform.ml.models import Embedder
    from sigplatform.ml.similarity_head import save_head

    train_p, val_p = load_split(os.path.join(data, "train")), load_split(os.path.join(data, "val"))
    if not train_p or not val_p:
        sys.exit("Need train/ and val/ folders with <person>/genuine images (see docstring).")
    print(f"people: train {len(train_p)}, val {len(val_p)}")

    emb = Embedder()                               # zero-shot DINOv2 (no head)
    ftr, fval = features(emb, train_p), features(emb, val_p)
    base = eval_pairs(val_p, normalised(fval))
    print("zero-shot DINOv2:", base)

    mlflow.set_tracking_uri(settings.mlflow_uri)
    mlflow.set_experiment(settings.experiment + "-similarity")
    os.makedirs(out_dir, exist_ok=True)
    best, best_path = None, os.path.join(out_dir, "head.pt")
    with mlflow.start_run(run_name="dinov2-triplet"):
        mlflow.log_params({"backbone": settings.dinov2_name, "licence": "Apache-2.0", "epochs": epochs,
                           "train_people": len(train_p), "val_people": len(val_p)})
        mlflow.log_metrics({f"zeroshot_{k}": v for k, v in base.items()})
        for ep, loss, head, in_dim, out_dim in train_head(train_p, ftr, epochs, device=emb.device):
            m = eval_pairs(val_p, normalised(fval, head))
            mlflow.log_metrics({"train_loss": loss, "val_eer": m["eer"], "val_roc_auc": m["roc_auc"]}, step=ep)
            print(f"epoch {ep:02d} loss {loss:.4f} EER {m['eer']:.4f} AUC {m['roc_auc']:.4f}")
            if best is None or m["eer"] < best["eer"]:
                best = m; save_head(head, best_path, in_dim, out_dim)
        mlflow.log_metrics({f"best_{k}": v for k, v in best.items()})
        version = registry.log_and_register(settings.similarity_model_name, {"head": best_path},
                                            {"val_eer": best["eer"], "match_threshold": best["threshold"]})

    cur, tags = registry.production_version(settings.similarity_model_name)
    if cur is None or best["eer"] < float(tags.get("val_eer", 1.0)):
        registry.promote(settings.similarity_model_name, version)
    print(json.dumps({"zero_shot": base, "fine_tuned": best, "version": version}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="signature_verification")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default=os.path.join("runs_similarity", "dinov2_head"))
    a = ap.parse_args()
    main(a.data, a.epochs, a.out)

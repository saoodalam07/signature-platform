"""Model wrappers + a manager that hot-reloads whatever is tagged @production in MLflow.

Detector  : RF-DETR Large  (Apache-2.0)  - finds signatures on a page
Similarity: DINOv2 ViT-B   (Apache-2.0)  - embeds a signature; cosine similarity compares two
"""
import logging
import os
import threading
import time

import numpy as np

from ..config import settings
from . import registry
from .imaging import normalise_signature

log = logging.getLogger(__name__)


class Detector:
    def __init__(self, weights=None, resolution=None):
        from rfdetr import RFDETRLarge
        kw = {"resolution": int(resolution)} if resolution else {}
        self.model = RFDETRLarge(pretrain_weights=weights, trust_checkpoint=True, **kw) if weights else RFDETRLarge()
        try:
            self.model.optimize_for_inference()
        except Exception as e:  # optional speed-up
            log.info("optimize_for_inference skipped: %s", e)

    def __call__(self, page, threshold):
        d = self.model.predict(page, threshold=threshold)
        return [(*map(float, box), float(c)) for box, c in zip(d.xyxy, d.confidence)]


class Embedder:
    def __init__(self, head_path=None):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(settings.dinov2_name)
        self.backbone = AutoModel.from_pretrained(settings.dinov2_name).to(self.device).eval()
        self.head = None
        if head_path:
            from .similarity_head import ProjectionHead
            state = torch.load(head_path, map_location=self.device)
            self.head = ProjectionHead(state["in_dim"], state["out_dim"]).to(self.device).eval()
            self.head.load_state_dict(state["weights"])

    def raw_features(self, images, batch=32):
        """DINOv2 features (CLS + mean patch token), not normalised. Shared by training and inference."""
        torch = self.torch
        feats = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                imgs = [normalise_signature(im) for im in images[i:i + batch]]
                x = self.proc(images=imgs, return_tensors="pt").to(self.device)
                out = self.backbone(**x).last_hidden_state
                feats.append(torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=1))
        return torch.cat(feats)

    def embed(self, images):
        torch = self.torch
        feat = self.raw_features(images)
        with torch.no_grad():
            if self.head is not None:
                feat = self.head(feat)
            feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy()


def cosine(a, b):
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


# ---------------------------------------------------------------- dummy models (tests, no GPU)
class DummyDetector:
    def __call__(self, page, threshold):
        w, h = page.size
        return [(w * 0.55, h * 0.75, w * 0.85, h * 0.85, 0.93)]


class DummyEmbedder:
    def embed(self, images):
        out = []
        for im in images:
            v = np.asarray(normalise_signature(im).convert("L").resize((16, 16)), dtype=np.float32).ravel()
            v = v - v.mean()
            out.append(v / (np.linalg.norm(v) + 1e-8))
        return np.stack(out)


# ---------------------------------------------------------------- manager
class ModelManager:
    """Loads models lazily and swaps them when the @production version changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._loaded = {}        # kind -> (version, model)
        self._checked = {}       # kind -> last check time
        self.tags = {}           # kind -> registry tags of the loaded version (e.g. match_threshold)

    def _load(self, kind, version, folder, tags=None):
        if settings.model_backend == "dummy":
            return DummyDetector() if kind == "detector" else DummyEmbedder()
        if kind == "detector":
            weights = registry.find_file(folder, "*.pth") if folder else os.environ.get("DETECTOR_WEIGHTS")
            if not weights:
                raise RuntimeError("No trained detector yet: run the training pipeline or set DETECTOR_WEIGHTS.")
            res = (tags or {}).get("resolution") or os.environ.get("DETECTOR_RESOLUTION")
            return Detector(weights, res)   # inference must use the training resolution
        head = registry.find_file(folder, "head.pt") if folder else None
        return Embedder(head)   # no fine-tuned head yet -> zero-shot DINOv2

    def get(self, kind):
        name = settings.detector_model_name if kind == "detector" else settings.similarity_model_name
        now = time.time()
        with self._lock:
            cur = self._loaded.get(kind)
            if cur and now - self._checked.get(kind, 0) < settings.model_check_seconds:
                return cur
            self._checked[kind] = now
            version, tags = None, {}
            if settings.model_backend != "dummy":
                version, tags = registry.production_version(name)
            if cur and cur[0] == version:
                return cur
            folder = registry.download(name, version) if version else None
            log.info("Loading %s version %s", kind, version or "fallback")
            model = self._load(kind, version, folder, tags)
            self._loaded[kind] = (version, model)
            self.tags[kind] = tags
            return self._loaded[kind]


models = ModelManager()

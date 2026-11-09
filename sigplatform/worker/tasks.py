"""Inference worker tasks (queue: "inference").

Start:  celery -A sigplatform.queue worker -Q inference -c 1 --loglevel=info
"""
import base64
import io
import time

from ..config import settings
from ..ml.imaging import crop, load_pages
from ..ml.models import cosine, models
from ..queue import celery_app
from ..storage import get_storage

import logfire
from celery.signals import worker_process_init


@worker_process_init.connect
def _init_logfire(**kwargs):
    # Start Logfire inside each Celery worker process
    logfire.configure(service_name="signature-worker", send_to_logfire="if-token-present")
    logfire.instrument_celery()
    logfire.instrument_system_metrics()


def _jpeg_b64(img, side=320):
    img = img.copy()
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _detect_pages(pages, threshold):
    version, det = models.get("detector")
    out = []
    for page in pages:
            with logfire.span("detect_page"):
                dets = det(page, threshold)
            out.append({"width": page.width, "height": page.height,
                    "detections": [{"box": [round(v, 1) for v in d[:4]], "confidence": round(d[4], 4)}
                                   for d in dets]})
    return version, out


def _read_and_delete(key):
    storage = get_storage()
    try:
        return storage.get(key)
    finally:
        storage.delete(key)        # privacy: uploads never outlive the job


@celery_app.task(name="sigplatform.worker.tasks.detect")
def detect(doc_key, threshold=None):
    t0 = time.perf_counter()
    with logfire.span("load_pages"):
        pages = load_pages(_read_and_delete(doc_key))
    version, result = _detect_pages(pages, threshold or settings.detect_threshold)
    return {"type": "detect", "model_version": version, "pages": result,
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


@celery_app.task(name="sigplatform.worker.tasks.similarity")
def similarity(sig_a_key, sig_b_key):
    """Compare two signature images directly."""
    from PIL import Image
    t0 = time.perf_counter()
    a = Image.open(io.BytesIO(_read_and_delete(sig_a_key))).convert("RGB")
    b = Image.open(io.BytesIO(_read_and_delete(sig_b_key))).convert("RGB")
    version, emb = models.get("similarity")
    ea, eb = emb.embed([a, b])
    score = cosine(ea, eb)
    thr = float(models.tags.get("similarity", {}).get("match_threshold", settings.match_threshold))
    return {"type": "similarity", "model_version": version or "dinov2-zero-shot",
            "score": round(score, 4), "threshold": thr, "match": score >= thr,
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


@celery_app.task(name="sigplatform.worker.tasks.verify")
def verify(doc_key, reference_key, threshold=None):
    """Detect every signature in a document and compare each with a reference signature."""
    from PIL import Image
    t0 = time.perf_counter()
    with logfire.span("load_pages"):
        pages = load_pages(_read_and_delete(doc_key))
    ref = Image.open(io.BytesIO(_read_and_delete(reference_key))).convert("RGB")
    det_version, result = _detect_pages(pages, threshold or settings.detect_threshold)

    crops, where = [], []
    for pi, (page, pr) in enumerate(zip(pages, result)):
        for di, d in enumerate(pr["detections"]):
            crops.append(crop(page, d["box"]))
            where.append((pi, di))

    sim_version, emb = models.get("similarity")
    thr = float(models.tags.get("similarity", {}).get("match_threshold", settings.match_threshold))
    best = None
    if crops:
        with logfire.span("embed", n_crops=len(crops)):
            vecs = emb.embed([ref] + crops)
        for (pi, di), c, v in zip(where, crops, vecs[1:]):
            s = round(cosine(vecs[0], v), 4)
            d = result[pi]["detections"][di]
            d.update({"similarity": s, "match": s >= thr, "crop": _jpeg_b64(c)})
            if best is None or s > best["similarity"]:
                best = {"page": pi, "index": di, "similarity": s, "match": s >= thr}
    return {"type": "verify", "detector_version": det_version, "similarity_version": sim_version or "dinov2-zero-shot",
            "threshold": thr, "best_match": best, "pages": result,
            "ms": round((time.perf_counter() - t0) * 1000, 1)}

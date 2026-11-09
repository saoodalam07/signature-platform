"""End-to-end test of API -> queue -> worker -> result, with dummy models (no GPU, no broker).

    MODEL_BACKEND=dummy python -m pytest -q tests
"""
import io
import os
import tempfile

os.environ.update({
    "MODEL_BACKEND": "dummy",
    "BROKER_URL": "memory://",
    "RESULT_BACKEND": "cache+memory://",
    "STORAGE_BACKEND": "local",
    "STORAGE_DIR": tempfile.mkdtemp(),
    "HF_WEBHOOK_SECRET": "test-secret",
    "HF_DATASET_REPO": "org/signature-dataset",
})

from fastapi.testclient import TestClient            # noqa: E402
from PIL import Image, ImageDraw                     # noqa: E402

from sigplatform.api import main as api               # noqa: E402
from sigplatform.queue import celery_app              # noqa: E402
import sigplatform.worker.tasks                       # noqa: E402,F401  (registers tasks)

celery_app.conf.task_always_eager = True
celery_app.conf.task_store_eager_result = True
sent = []


def eager_send(name, *args):
    sent.append(name)
    if name.startswith("sigplatform.trainer"):
        class R: id = "train-job-1"
        return R()
    return celery_app.tasks[name].apply(args=list(args))


api._send = lambda name, *a, **opts: eager_send(f"sigplatform.{name}", *a)
client = TestClient(api.app)


def png(draw_sig=True, seed=0):
    img = Image.new("RGB", (800, 1000), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 60), "INVOICE #123", fill="black")
    if draw_sig:
        pts = [(460 + i * 12, 790 + ((i * 37 + seed) % 40)) for i in range(20)]
        d.line(pts, fill="navy", width=4)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def result(resp):
    assert resp.status_code == 202, resp.text
    job = client.get(resp.json()["status_url"]).json()
    assert job["status"] == "done", job
    return job["result"]


def test_detect():
    r = result(client.post("/v1/detect", files={"document": ("doc.png", png(), "image/png")}))
    assert r["type"] == "detect" and len(r["pages"][0]["detections"]) == 1


def test_similarity_same_vs_different():
    same = result(client.post("/v1/similarity", files={"signature_a": ("a.png", png(seed=0), "image/png"),
                                                        "signature_b": ("b.png", png(seed=0), "image/png")}))
    assert same["score"] > 0.99 and same["match"]


def test_verify():
    r = result(client.post("/v1/verify", files={"document": ("doc.png", png(), "image/png"),
                                                "reference": ("ref.png", png(), "image/png")}))
    assert r["best_match"] is not None and "similarity" in r["pages"][0]["detections"][0]


def test_uploads_are_deleted():
    assert os.listdir(os.environ["STORAGE_DIR"]) == []


def test_rejects_bad_file():
    r = client.post("/v1/detect", files={"document": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_webhook_auth_and_trigger():
    body = {"event": {"action": "update", "scope": "repo.content"},
            "repo": {"type": "dataset", "name": "org/signature-dataset", "headSha": "abc123"}}
    assert client.post("/webhooks/huggingface", json=body).status_code == 401
    r = client.post("/webhooks/huggingface", json=body, headers={"X-Webhook-Secret": "test-secret"})
    assert r.status_code == 202 and r.json()["status"] == "training_queued"
    assert sent[-1] == "sigplatform.trainer.tasks.train_on_new_data"
    other = dict(body, repo={"type": "model", "name": "x"})
    assert client.post("/webhooks/huggingface", json=other,
                       headers={"X-Webhook-Secret": "test-secret"}).json()["status"] == "ignored"

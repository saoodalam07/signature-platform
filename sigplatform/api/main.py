"""Public FastAPI service. It never runs a model itself: it validates input, stores the
upload, puts a job on the queue and returns a job id. Workers do the heavy lifting.

Run:  uvicorn sigplatform.api.main:app --host 0.0.0.0 --port 8000

Endpoints
  POST /v1/detect                 document                       -> job id
  POST /v1/verify                 document + reference signature -> job id
  POST /v1/similarity             signature A + signature B      -> job id
  GET  /v1/jobs/{job_id}          status + result
  POST /webhooks/huggingface      HF dataset changed -> start remote training
  GET  /v1/models                 what is in production now
  GET  /health
"""
import hmac
import logging

from celery.result import AsyncResult
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile

from ..config import settings
from ..queue import celery_app
from ..storage import get_storage

log = logging.getLogger("api")
app = FastAPI(title="Signature Platform API", version="1.0.0")

ALLOWED = {b"%PDF": ".pdf", b"\x89PNG": ".png", b"\xff\xd8\xff": ".jpg"}


async def _store(f: UploadFile) -> str:
    data = await f.read(settings.max_bytes + 1)
    if not data:
        raise HTTPException(400, f"{f.filename or 'file'} is empty.")
    if len(data) > settings.max_bytes:
        raise HTTPException(413, f"{f.filename} is larger than {settings.max_bytes // 2**20} MB.")
    suffix = next((ext for magic, ext in ALLOWED.items() if data.startswith(magic)), None)
    if suffix is None:
        raise HTTPException(415, f"{f.filename}: use JPG, PNG or PDF.")
    return get_storage().put(data, suffix)


def _send(name, *args, **options):
    # by name: the API image does not need torch or any model code
    return celery_app.send_task(f"sigplatform.{name}", args=list(args), **options)


def _queued(task):
    return {"job_id": task.id, "status": "queued", "status_url": f"/v1/jobs/{task.id}"}


@app.post("/v1/detect", status_code=202)
async def detect(document: UploadFile = File(...), threshold: float | None = None):
    return _queued(_send("worker.tasks.detect", await _store(document), threshold))


@app.post("/v1/verify", status_code=202)
async def verify(document: UploadFile = File(...), reference: UploadFile = File(...),
                 threshold: float | None = None):
    return _queued(_send("worker.tasks.verify", await _store(document), await _store(reference), threshold))


@app.post("/v1/similarity", status_code=202)
async def similarity(signature_a: UploadFile = File(...), signature_b: UploadFile = File(...)):
    return _queued(_send("worker.tasks.similarity", await _store(signature_a), await _store(signature_b)))


@app.get("/v1/jobs/{job_id}")
def job(job_id: str):
    r = AsyncResult(job_id, app=celery_app)
    if r.state in ("PENDING", "RECEIVED"):
        return {"job_id": job_id, "status": "queued"}
    if r.state == "STARTED":
        return {"job_id": job_id, "status": "running"}
    if r.state == "FAILURE":
        return {"job_id": job_id, "status": "failed", "error": str(r.result)}
    return {"job_id": job_id, "status": "done", "result": r.result}


@app.post("/webhooks/huggingface", status_code=202)
async def huggingface_webhook(request: Request, x_webhook_secret: str | None = Header(default=None)):
    """Hugging Face calls this when the dataset repo changes (Settings -> Webhooks)."""
    if not settings.hf_webhook_secret or not x_webhook_secret or \
            not hmac.compare_digest(x_webhook_secret, settings.hf_webhook_secret):
        raise HTTPException(401, "Invalid webhook secret.")
    body = await request.json()
    event, repo = body.get("event", {}), body.get("repo", {})
    if repo.get("type") != "dataset" or event.get("scope") != "repo.content" or event.get("action") != "update":
        return {"status": "ignored", "reason": "not a dataset content update"}
    if settings.hf_dataset_repo and repo.get("name") != settings.hf_dataset_repo:
        return {"status": "ignored", "reason": f"repo {repo.get('name')} is not watched"}
    # A large upload arrives as several commits = several webhooks. Queue ONE job, delayed a little,
    # that trains on the latest revision; ignore further webhooks while that job is pending.
    if not _claim_training_slot():
        return {"status": "ignored", "reason": "a training job is already queued; it will use the latest data"}
    task = _send("trainer.tasks.train_on_new_data", repo.get("name"), None, countdown=TRAIN_DEBOUNCE_S)
    log.info("Training job %s queued for %s (starts in %ss, latest revision)", task.id, repo.get("name"),
             TRAIN_DEBOUNCE_S)
    return {"status": "training_queued", "job_id": task.id, "starts_in_s": TRAIN_DEBOUNCE_S}


TRAIN_DEBOUNCE_S = 180


def _claim_training_slot():
    """True for the first webhook in a burst; False while a job from that burst is still pending."""
    try:
        import redis
        r = redis.Redis.from_url(settings.result_backend, socket_timeout=2)
        return bool(r.set("sigplatform:training-pending", "1", nx=True, ex=TRAIN_DEBOUNCE_S + 60))
    except Exception as e:  # no Valkey available: never block training
        log.warning("Webhook de-duplication unavailable (%s); queuing anyway", e)
        return True


@app.get("/v1/models")
def deployed_models():
    if settings.model_backend == "dummy":
        return {"detector": "dummy", "similarity": "dummy"}
    from ..ml.registry import production_version
    out = {}
    for kind, name in (("detector", settings.detector_model_name), ("similarity", settings.similarity_model_name)):
        v, tags = production_version(name)
        out[kind] = {"model": name, "version": v, **tags}
    return out


@app.get("/health")
def health():
    return {"status": "ok"}

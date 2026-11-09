"""Celery app: one broker, two queues.

  inference  -> CPU/GPU inference workers (detect, verify, similarity)
  training   -> remote GPU trainer (one job at a time)
"""
from celery import Celery

from .config import settings

broker_options = {}
if settings.broker_url.startswith("sqs://"):
    # visibility timeout > longest training job, so SQS does not hand the same job to a second trainer
    broker_options = {"region": settings.sqs_region, "visibility_timeout": 12 * 3600, "polling_interval": 1,
                      "queue_name_prefix": "sigplatform-"}

# RESULT_BACKEND=disabled: the remote laptop trainer does not need (or get) access to the results store
backend = None if settings.result_backend == "disabled" else settings.result_backend
celery_app = Celery("sigplatform", broker=settings.broker_url, backend=backend)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_routes={
        "sigplatform.worker.tasks.*": {"queue": "inference"},
        "sigplatform.trainer.tasks.*": {"queue": "training"},
    },
    task_acks_late=True,               # a crashed worker does not lose the job
    worker_prefetch_multiplier=1,      # fair dispatch for long GPU jobs
    result_expires=24 * 3600,
    broker_transport_options=broker_options,
    include=["sigplatform.worker.tasks", "sigplatform.trainer.tasks"],
)

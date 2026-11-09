"""Central settings, read from environment variables (see .env.example)."""
import os


def _env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


class Settings:
    # Queue (Celery). Local: Valkey (BSD, Redis-compatible). AWS: "sqs://" (uses IAM role).
    broker_url = _env("BROKER_URL", "redis://localhost:6379/0")
    result_backend = _env("RESULT_BACKEND", "redis://localhost:6379/1")
    sqs_region = _env("AWS_REGION", "us-east-1")

    # Object storage for uploaded documents. "local" (shared volume) or "s3".
    storage_backend = _env("STORAGE_BACKEND", "local")
    storage_dir = _env("STORAGE_DIR", "/data/uploads")
    s3_bucket = _env("S3_BUCKET")

    # MLflow tracking + model registry
    mlflow_uri = _env("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment = _env("MLFLOW_EXPERIMENT", "signature-detection")
    detector_model_name = _env("DETECTOR_MODEL_NAME", "signature-detector")
    similarity_model_name = _env("SIMILARITY_MODEL_NAME", "signature-similarity")
    production_alias = _env("PRODUCTION_ALIAS", "production")
    model_cache_dir = _env("MODEL_CACHE_DIR", "/models")
    model_check_seconds = int(_env("MODEL_CHECK_SECONDS", "60"))

    # Inference
    model_backend = _env("MODEL_BACKEND", "real")          # "real" or "dummy" (tests)
    detect_threshold = float(_env("DETECT_THRESHOLD", "0.40"))
    match_threshold = float(_env("MATCH_THRESHOLD", "0.85"))  # cosine similarity for "same person"
    dinov2_name = _env("DINOV2_NAME", "facebook/dinov2-base")
    max_bytes = int(_env("MAX_UPLOAD_MB", "15")) * 1024 * 1024
    max_pages = int(_env("MAX_PAGES", "5"))

    # Training pipeline
    hf_webhook_secret = _env("HF_WEBHOOK_SECRET")
    hf_dataset_repo = _env("HF_DATASET_REPO")              # e.g. "sevenminds/signature-dataset"
    hf_token = _env("HF_TOKEN")
    train_epochs = int(_env("TRAIN_EPOCHS", "50"))
    train_batch = int(_env("TRAIN_BATCH", "4"))
    train_resolution = int(_env("TRAIN_RESOLUTION", "640"))   # RF-DETR L default 704; 640 trains ~20% faster
    train_patience = int(_env("TRAIN_PATIENCE", "4"))
    work_dir = _env("TRAIN_WORK_DIR", "/work")
    promote_min_gain = float(_env("PROMOTE_MIN_GAIN", "0.0"))  # new mAP50 must beat production by this


settings = Settings()

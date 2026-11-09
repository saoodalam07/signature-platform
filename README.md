# Signature Platform (NETSOL, confidential)

Signature **detection** + signature **similarity** as a production service, with an automatic
**retraining pipeline**. Every component is MIT / Apache / BSD licensed. **No Ultralytics (AGPL).**

```
Client ──► FastAPI (api) ──► Queue (Valkey / AWS SQS) ──► Inference worker ──► result
                 ▲                                         │ loads models:/…@production
                 │                                         ▼
 Hugging Face dataset push ──webhook──► training queue ──► GPU trainer ──► MLflow registry
                                                         (RF-DETR Large)   (promote if better)
```

## What is in here

| Requirement | Where |
|---|---|
| Object detection for signatures | RF-DETR **Large** (Apache-2.0) · `training/train_detector.py`, `sigplatform/ml/models.py` |
| Signature similarity score | DINOv2 ViT-B (Apache-2.0) + triplet-trained head · `training/train_similarity.py` |
| FastAPI | `sigplatform/api/main.py` |
| Worker service | `sigplatform/worker/tasks.py` (inference), `sigplatform/trainer/tasks.py` (training) |
| Message + queue system | Celery on Valkey (local) or **AWS SQS** (prod) · `sigplatform/queue.py` |
| Deployed end to end | `docker-compose.yml`, `docker/*.Dockerfile`, AWS section below |
| Architecture diagram (AWS style) | `docs/architecture.html` |
| Training pipeline on HF upload | `/webhooks/huggingface` → `training/pipeline.py` |
| Tracking | **MLflow** (metrics, artifacts, model registry, `@production` alias) |

## API

| Method | Path | Body (multipart) | Returns |
|---|---|---|---|
| POST | `/v1/detect` | `document` (JPG/PNG/PDF) | job id |
| POST | `/v1/verify` | `document` + `reference` (signature image) | job id |
| POST | `/v1/similarity` | `signature_a` + `signature_b` | job id |
| GET | `/v1/jobs/{id}` | — | `queued / running / done / failed` + result |
| GET | `/v1/models` | — | model versions in production |
| POST | `/webhooks/huggingface` | HF webhook JSON (`X-Webhook-Secret`) | training job id |

Interactive docs: `http://localhost:8000/docs`

Example:
```bash
curl -F document=@cheque.pdf -F reference=@ref_signature.png http://localhost:8000/v1/verify
curl http://localhost:8000/v1/jobs/<job_id>
```
`verify` returns every detected signature with a box, detection confidence, similarity score
(0–1), match yes/no and a small crop, plus the best match.

## 1. Train the first detector (your RTX A3000 laptop)

```bat
python -m venv rfdetr_env
rfdetr_env\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements\trainer.txt

docker compose up -d valkey mlflow          :: MLflow UI -> http://localhost:5000
set MLFLOW_TRACKING_URI=http://localhost:5000
python -m training.train_detector --data C:\Users\DELL\Downloads\signature_dataset_coco
```
- Out of GPU memory → `--batch 2`. Stopped? → add `--resume`.
- At the end it evaluates on the **test** split (mAP50, mAP50-95, precision, recall, F1, ms/image),
  registers `signature-detector` in MLflow and promotes it to `@production`.

## 2. Similarity model

Works immediately with **zero-shot DINOv2** (no training). For better genuine-vs-forged separation,
train the head on a verification dataset (one folder per person, `genuine/` and `forged/`):
```bat
python -m training.train_similarity --data signature_verification
```
Logs zero-shot vs fine-tuned **EER / ROC-AUC** and the best match threshold to MLflow, and
promotes `signature-similarity@production` if EER improves.
CEDAR / BHSig260 / SigComp have this structure — check each licence before production use.

## 3. Run the whole stack

```bash
cp .env.example .env              # edit secrets
docker compose up -d --build      # api, worker, valkey, mlflow
docker compose --profile gpu up -d trainer   # GPU training worker (NVIDIA Container Toolkit)
```
Windows without Docker for workers: add `--pool=solo` to the celery command.

## 4. Automatic retraining from Hugging Face

1. Upload the dataset to a **private** HF dataset repo (each upload = new commit):
   `python -m training.push_dataset_to_hf --src signature_dataset_coco --repo your-org/signature-dataset`
2. HF → repo **Settings → Webhooks** → URL `https://<your-api>/webhooks/huggingface`,
   secret = `HF_WEBHOOK_SECRET`, events: **Repo update**.
3. On every push: API checks the secret → queues `train_on_new_data` → the GPU trainer downloads that
   exact revision → trains RF-DETR Large → evaluates on test → registers in MLflow → **promotes only if
   mAP50 ≥ production** → inference workers pick up the new `@production` version within
   `MODEL_CHECK_SECONDS` (default 60 s). No restart, no downtime.

The API must be reachable from the internet for the webhook (AWS deploy, or a tunnel while testing).

## 5. AWS deployment (production)

| Local | AWS |
|---|---|
| api container | ECS Fargate service behind an Application Load Balancer (HTTPS via ACM) |
| Valkey broker | **Amazon SQS** (`BROKER_URL=sqs://`, IAM role) + ElastiCache (Valkey) for job results |
| uploads volume | **S3** bucket (`STORAGE_BACKEND=s3`, SSE, lifecycle delete after 1 day) |
| worker container | ECS on EC2 GPU (g5/g6) or Fargate CPU, autoscaled on SQS queue depth |
| trainer container | EC2 GPU instance (g5.xlarge), started on demand |
| MLflow | ECS service + RDS PostgreSQL (backend) + S3 (artifacts) |
| logs / metrics | CloudWatch |
| images | ECR |
| secrets | Secrets Manager (HF token, webhook secret) |

## Privacy

Uploads are written to temporary storage only for the job and deleted by the worker right after
processing. The dataset repo on Hugging Face must stay **private** (NETSOL confidential).

## Tests

```bash
pip install fastapi celery python-multipart httpx pytest pillow numpy
python -m pytest -q tests        # API → queue → worker → result with dummy models
```

## Licences

See `LICENSES.md`. Do **not** install `rfdetr[plus]` (XL/2XL use the PML licence) and do not add
Ultralytics or PyMuPDF (AGPL).

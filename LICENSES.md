# Third-party licences

Every runtime dependency is permissive (MIT / Apache-2.0 / BSD). Verify again before release.

| Component | Used for | Licence |
|---|---|---|
| RF-DETR (`rfdetr`), **Large** weights | signature detection | Apache-2.0 |
| DINOv2 (`facebook/dinov2-base`) | signature embeddings | Apache-2.0 |
| PyTorch, torchvision | deep learning | BSD-3-Clause |
| Hugging Face Transformers, huggingface_hub | DINOv2, dataset download | Apache-2.0 |
| PyTorch Lightning (via rfdetr[train]) | training loop | Apache-2.0 |
| FastAPI | API | MIT |
| Uvicorn | ASGI server | BSD-3-Clause |
| Celery, Kombu | worker + queue | BSD-3-Clause |
| Valkey | broker / results (local) | BSD-3-Clause |
| MLflow | tracking + registry | Apache-2.0 |
| pypdfium2 | PDF rendering | Apache-2.0 / BSD-3-Clause |
| Pillow | images | MIT-CMU (HPND) |
| NumPy, scikit-learn, pycocotools | maths / metrics | BSD |
| boto3 | AWS (S3, SQS) | Apache-2.0 |

## Deliberately NOT used

| Component | Why |
|---|---|
| Ultralytics YOLO (v5/v8/11/26) | AGPL-3.0 |
| RF-DETR XL / 2XL (`rfdetr[plus]`) | Roboflow PML 1.0 |
| PyMuPDF | AGPL-3.0 (replaced by pypdfium2) |
| Redis 8 | RSAL / SSPL / AGPL (replaced by Valkey) |
| DINOv3 | custom Meta licence, needs legal review |

Apache-2.0 requires shipping the licence text and any NOTICE files with the product.

r"""Register an already-trained detector checkpoint in a (new) MLflow server and promote it.

Used once when moving to AWS: copies your laptop's v1 model into the cloud MLflow registry,
so the cloud workers serve exactly the model you tested.

    set MLFLOW_TRACKING_URI=http://<EC2-IP>:5000
    python -m training.register_existing --weights runs_rfdetr\signature_rfdetr_large\checkpoint_best_total.pth ^
        --map50 0.9649 --map50-95 0.7344 --resolution 576
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sigplatform.config import settings     # noqa: E402
from sigplatform.ml import registry         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--map50", type=float, required=True)
ap.add_argument("--map50-95", type=float, required=True)
ap.add_argument("--resolution", type=int, default=576)
ap.add_argument("--no-promote", action="store_true")
a = ap.parse_args()
if not os.path.exists(a.weights):
    sys.exit(f"Not found: {a.weights}")

import mlflow  # noqa: E402

mlflow.set_tracking_uri(settings.mlflow_uri)
mlflow.set_experiment(settings.experiment)
with mlflow.start_run(run_name="imported-from-laptop"):
    mlflow.log_params({"model": "rfdetr-large", "licence": "Apache-2.0", "resolution": a.resolution,
                       "source": "laptop training (imported)"})
    mlflow.log_metrics({"test_map50": a.map50, "test_map50_95": a.map50_95})
    version = registry.log_and_register(
        settings.detector_model_name, {"weights": a.weights},
        {"test_map50": a.map50, "test_map50_95": a.map50_95, "arch": "rfdetr-large",
         "resolution": a.resolution, "dataset_revision": "local"})
if not a.no_promote:
    registry.promote(settings.detector_model_name, version)
print(f"Registered {settings.detector_model_name} v{version}" + ("" if a.no_promote else " and promoted to @production"))

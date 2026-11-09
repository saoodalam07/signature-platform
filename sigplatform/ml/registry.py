"""MLflow Model Registry helpers.

The "production" alias on a registered model is the single source of truth for what is deployed.
The training pipeline moves the alias; inference workers notice the change and hot-reload.
"""
import glob
import logging
import os

from ..config import settings

log = logging.getLogger(__name__)


def _client():
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(settings.mlflow_uri)
    return MlflowClient()


def production_version(name):
    """Return (version, tags) for name@production, or (None, {}) if nothing is deployed yet."""
    try:
        mv = _client().get_model_version_by_alias(name, settings.production_alias)
        return str(mv.version), dict(mv.tags or {})
    except Exception as e:  # registry empty / MLflow down
        log.info("No production version for %s (%s)", name, e)
        return None, {}


def download(name, version):
    """Download a registered model version and return the local folder."""
    import mlflow
    dst = os.path.join(settings.model_cache_dir, name, str(version))
    if not os.path.isdir(dst) or not os.listdir(dst):
        os.makedirs(dst, exist_ok=True)
        mlflow.set_tracking_uri(settings.mlflow_uri)
        mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{version}", dst_path=dst)
    return dst


def find_file(folder, pattern):
    hits = sorted(glob.glob(os.path.join(folder, "**", pattern), recursive=True))
    return hits[0] if hits else None


class _ArtifactOnly:
    """Minimal pyfunc wrapper so plain weight files can live in the Model Registry."""

    def __new__(cls):
        import mlflow.pyfunc

        class Wrapper(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input, params=None):
                raise NotImplementedError("Load the weights from context.artifacts in the worker.")
        return Wrapper()


def log_and_register(name, artifacts: dict, tags: dict):
    """Log weight files as a model in the active MLflow run and register a new version.

    Returns the new version number (str). Works with MLflow 2.x and 3.x.
    """
    import mlflow
    try:
        info = mlflow.pyfunc.log_model(name="model", python_model=_ArtifactOnly(), artifacts=artifacts)
    except TypeError:  # MLflow 2.x
        info = mlflow.pyfunc.log_model(artifact_path="model", python_model=_ArtifactOnly(), artifacts=artifacts)
    mv = mlflow.register_model(info.model_uri, name)
    client = _client()
    for k, v in tags.items():
        client.set_model_version_tag(name, mv.version, k, str(v))
    return str(mv.version)


def promote(name, version):
    _client().set_registered_model_alias(name, settings.production_alias, str(version))
    log.info("Promoted %s v%s to @%s", name, version, settings.production_alias)

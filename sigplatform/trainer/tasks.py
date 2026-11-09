"""Remote training worker (queue: "training"). Runs on a GPU machine: your RTX A3000 laptop,
an AWS EC2 g5/g6 instance, or any box that can reach the broker and MLflow.

Start:  celery -A sigplatform.queue worker -Q training -c 1 --loglevel=info
"""
from ..queue import celery_app


@celery_app.task(name="sigplatform.trainer.tasks.train_on_new_data", bind=True, acks_late=True,
                 ignore_result=True, time_limit=11 * 3600, soft_time_limit=10 * 3600)
def train_on_new_data(self, repo, revision=None):
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:      # the `celery` launcher does not add the project folder
        sys.path.insert(0, root)
    from training.pipeline import run
    return run(repo, revision)

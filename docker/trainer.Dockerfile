FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
WORKDIR /app
COPY requirements requirements
RUN pip install --no-cache-dir -r requirements/trainer.txt
COPY sigplatform sigplatform
COPY training training
CMD ["celery", "-A", "sigplatform.queue", "worker", "-Q", "training", "-c", "1", "--loglevel=info"]

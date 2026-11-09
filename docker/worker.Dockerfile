FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY requirements/worker.txt requirements/worker.txt
RUN pip install --no-cache-dir -r requirements/worker.txt
COPY sigplatform sigplatform
CMD ["celery", "-A", "sigplatform.queue", "worker", "-Q", "inference", "-c", "1", "--loglevel=info"]

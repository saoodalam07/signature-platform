FROM python:3.11-slim
WORKDIR /app
# OpenCV (pulled in by rfdetr -> supervision) needs these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# CPU build of a current PyTorch (transformers 5.x needs torch >= 2.6). On GPU hosts use the CUDA index instead.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements/worker.txt requirements/worker.txt
RUN pip install --no-cache-dir -r requirements/worker.txt
COPY sigplatform sigplatform
CMD ["celery", "-A", "sigplatform.queue", "worker", "-Q", "inference", "-c", "1", "--loglevel=info"]

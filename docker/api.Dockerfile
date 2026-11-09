FROM python:3.11-slim
WORKDIR /app
COPY requirements/api.txt requirements/api.txt
RUN pip install --no-cache-dir -r requirements/api.txt
COPY sigplatform sigplatform
RUN useradd -m app && mkdir -p /data/uploads && chown app /data/uploads
USER app
EXPOSE 8000
CMD ["uvicorn", "sigplatform.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

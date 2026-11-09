"""Temporary object storage for uploads.

The API saves the upload, puts only a small key on the queue (SQS messages are limited
to 256 KB), and the worker reads and then deletes it. Nothing is kept after processing.
"""
import os
import uuid

from .config import settings


class LocalStorage:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def put(self, data: bytes, suffix="") -> str:
        key = f"{uuid.uuid4().hex}{suffix}"
        with open(os.path.join(self.root, key), "wb") as f:
            f.write(data)
        return key

    def get(self, key: str) -> bytes:
        with open(os.path.join(self.root, os.path.basename(key)), "rb") as f:
            return f.read()

    def delete(self, key: str):
        try:
            os.remove(os.path.join(self.root, os.path.basename(key)))
        except FileNotFoundError:
            pass


class S3Storage:
    def __init__(self, bucket):
        import boto3
        self.bucket, self.s3 = bucket, boto3.client("s3")

    def put(self, data: bytes, suffix="") -> str:
        key = f"uploads/{uuid.uuid4().hex}{suffix}"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ServerSideEncryption="AES256")
        return key

    def get(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def delete(self, key: str):
        self.s3.delete_object(Bucket=self.bucket, Key=key)


_storage = None


def get_storage():
    global _storage
    if _storage is None:
        _storage = S3Storage(settings.s3_bucket) if settings.storage_backend == "s3" else LocalStorage(settings.storage_dir)
    return _storage

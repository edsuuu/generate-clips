from __future__ import annotations

import hashlib
import mimetypes
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

from app.storage.base import StoredFile
from app.support.config import settings


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class MinioStorageProvider:
    def __init__(self):
        from minio import Minio

        parsed = urlparse(settings.minio_endpoint)
        endpoint = parsed.netloc or parsed.path
        secure = settings.minio_secure or parsed.scheme == "https"
        self.bucket = settings.minio_bucket
        self.client = Minio(
            endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=secure,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def upload_file(
        self,
        local_path: Path,
        object_path: str,
        file_type: str,
        content_type: str | None = None,
    ) -> StoredFile:
        content_type = content_type or mimetypes.guess_type(local_path.name)[0]
        self.client.fput_object(
            self.bucket,
            object_path,
            str(local_path),
            content_type=content_type or "application/octet-stream",
        )

        return StoredFile(
            type=file_type,
            disk="minio",
            bucket=self.bucket,
            path=object_path,
            mime_type=content_type,
            extension=local_path.suffix.lstrip(".") or None,
            size_bytes=local_path.stat().st_size,
            checksum_sha256=_sha256(local_path),
        )

    def download_file(self, object_path: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.fget_object(self.bucket, object_path, str(local_path))
        return local_path

    def presigned_get_url(self, object_path: str, expires_seconds: int = 3600) -> str:
        return self.client.presigned_get_object(
            self.bucket,
            object_path,
            expires=timedelta(seconds=expires_seconds),
        )

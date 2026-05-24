from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class StoredFile:
    type: str
    disk: str
    bucket: str
    path: str
    mime_type: str | None = None
    extension: str | None = None
    size_bytes: int | None = None
    checksum_sha256: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class StorageProvider(Protocol):
    def upload_file(
        self,
        local_path: Path,
        object_path: str,
        file_type: str,
        content_type: str | None = None,
    ) -> StoredFile:
        ...

    def download_file(self, object_path: str, local_path: Path) -> Path:
        ...

    def presigned_get_url(self, object_path: str, expires_seconds: int = 3600) -> str:
        ...

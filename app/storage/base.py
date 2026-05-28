from __future__ import annotations

from dataclasses import asdict, dataclass


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

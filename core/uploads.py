"""Durable, resumable upload sessions for the Workbench HTTP service."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from core.workflows import IMAGE_SUFFIXES, ROOT, VIDEO_SUFFIXES


UPLOAD_ROOT = ROOT / "datasets" / ".upload-sessions"
DEFAULT_MAX_BYTES = 20 * 1024 * 1024 * 1024
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_RETENTION_SECONDS = 7 * 24 * 60 * 60
UPLOAD_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)

KIND_EXTENSIONS = {
    "dataset": {".zip"},
    "model": {".pt"},
    "image": set(IMAGE_SUFFIXES),
    "video": set(VIDEO_SUFFIXES),
}


class UploadError(ValueError):
    """Expected client-facing upload error."""

    status_code = 422


class UploadNotFound(UploadError):
    status_code = 404


class UploadConflict(UploadError):
    status_code = 409


class UploadExpired(UploadError):
    status_code = 410


class UploadTooLarge(UploadError):
    status_code = 413


class UploadStorageError(OSError):
    status_code = 507


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def max_upload_bytes() -> int:
    return _env_int("YOLOV10_UPLOAD_MAX_BYTES", DEFAULT_MAX_BYTES, 1)


def upload_chunk_bytes() -> int:
    return _env_int("YOLOV10_UPLOAD_CHUNK_BYTES", DEFAULT_CHUNK_BYTES, 1)


def upload_retention_seconds() -> int:
    return _env_int("YOLOV10_UPLOAD_RETENTION_SECONDS", DEFAULT_RETENTION_SECONDS, 60)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


class UploadStore:
    """Persist upload metadata and bytes so a request or process restart is recoverable."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root or UPLOAD_ROOT)
        self._lock = threading.RLock()

    def _session_dir(self, upload_id: str) -> Path:
        if not UPLOAD_ID_PATTERN.fullmatch(str(upload_id)):
            raise UploadNotFound("Upload session not found.")
        return self.root / upload_id

    def _metadata_path(self, upload_id: str) -> Path:
        return self._session_dir(upload_id) / "meta.json"

    def _payload_path(self, upload_id: str) -> Path:
        return self._session_dir(upload_id) / "payload.part"

    def _read(self, upload_id: str) -> Dict[str, Any]:
        directory = self._session_dir(upload_id)
        metadata = directory / "meta.json"
        if not metadata.is_file():
            raise UploadNotFound("Upload session not found.")
        try:
            value = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UploadConflict("Upload session metadata is corrupted.") from exc
        if not isinstance(value, dict):
            raise UploadConflict("Upload session metadata is corrupted.")
        if value.get("status") == "expired":
            raise UploadExpired("Upload session expired; start the upload again.")
        return value

    def _write(self, upload_id: str, value: Dict[str, Any]) -> None:
        directory = self._session_dir(upload_id)
        temporary = directory / "meta.json.tmp"
        with temporary.open("w", encoding="utf-8") as target:
            target.write(json.dumps(value, ensure_ascii=False, indent=2))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, directory / "meta.json")

    @staticmethod
    def _public(value: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value[key]
            for key in (
                "upload_id",
                "kind",
                "filename",
                "size",
                "sha256",
                "chunk_size",
                "offset",
                "status",
                "created_at",
                "updated_at",
                "result",
                "error",
            )
            if key in value
        }

    def create(self, kind: str, filename: str, size: int, sha256: str) -> Dict[str, Any]:
        kind = str(kind or "").strip().lower()
        incoming = Path(str(filename or "")).name
        suffix = Path(incoming).suffix.lower()
        if kind not in KIND_EXTENSIONS:
            raise UploadError("Unsupported upload type.")
        if not incoming or incoming in {".", ".."} or "\x00" in incoming:
            raise UploadError("Uploaded file needs a valid filename.")
        if suffix not in KIND_EXTENSIONS[kind]:
            if kind == "dataset":
                raise UploadError("Only .zip dataset archives are supported.")
            if kind == "model":
                raise UploadError("Only .pt model files are supported.")
            raise UploadError(f"Unsupported {kind} file format.")
        try:
            size = int(size)
        except (TypeError, ValueError) as exc:
            raise UploadError("Upload size must be an integer.") from exc
        if size < 1:
            raise UploadError("Uploaded files cannot be empty.")
        if size > max_upload_bytes():
            raise UploadTooLarge(f"The upload exceeds the {max_upload_bytes() / (1024 ** 3):.0f} GB limit.")
        sha256 = str(sha256 or "").strip().lower()
        if not SHA256_PATTERN.fullmatch(sha256):
            raise UploadError("A SHA-256 checksum is required before uploading.")

        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            upload_id = uuid.uuid4().hex
            directory = self._session_dir(upload_id)
            directory.mkdir(mode=0o700)
            metadata = {
                "upload_id": upload_id,
                "kind": kind,
                "filename": incoming,
                "size": size,
                "sha256": sha256,
                "chunk_size": upload_chunk_bytes(),
                "offset": 0,
                "status": "uploading",
                "created_at": _iso(_now()),
                "updated_at": _iso(_now()),
            }
            try:
                # A sparse file gives an early, stable target for offset writes without
                # allocating the whole 20 GB limit up front.
                with (directory / "payload.part").open("wb") as payload:
                    payload.truncate(size)
                self._write(upload_id, metadata)
            except OSError as exc:
                shutil.rmtree(directory, ignore_errors=True)
                raise UploadStorageError("The server could not create an upload session.") from exc
        return self._public(metadata)

    def get(self, upload_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._public(self._read(upload_id))

    def append(self, upload_id: str, offset: int, payload: bytes) -> Dict[str, Any]:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise UploadError("Upload body must contain raw file bytes.")
        data = bytes(payload)
        if len(data) > upload_chunk_bytes():
            raise UploadTooLarge("Upload chunk is larger than the configured chunk size.")
        try:
            offset = int(offset)
        except (TypeError, ValueError) as exc:
            raise UploadError("Upload-Offset must be an integer.") from exc
        if offset < 0:
            raise UploadError("Upload-Offset cannot be negative.")
        with self._lock:
            metadata = self._read(upload_id)
            if metadata["status"] != "uploading":
                if metadata["status"] == "completed" and offset == metadata["size"]:
                    return self._public(metadata)
                raise UploadConflict("Upload session is no longer accepting chunks.")
            current = int(metadata["offset"])
            expected_size = int(metadata["size"])
            if offset > current:
                raise UploadConflict(f"Upload offset is {current}; resume from that offset.")
            if offset + len(data) > expected_size:
                raise UploadError("Upload chunk exceeds the declared file size.")
            payload_path = self._payload_path(upload_id)
            try:
                if offset < current:
                    # A lost response can cause a client retry. Treat an identical
                    # already-written chunk as idempotent instead of rejecting it.
                    with payload_path.open("rb") as existing:
                        existing.seek(offset)
                        if existing.read(len(data)) != data:
                            raise UploadConflict("A retried chunk does not match the stored bytes.")
                elif data:
                    with payload_path.open("r+b") as target:
                        target.seek(offset)
                        written = target.write(data)
                        if written != len(data):
                            raise UploadStorageError("The server could not save the complete upload chunk.")
                        target.flush()
                        os.fsync(target.fileno())
                    metadata["offset"] = offset + len(data)
                metadata["updated_at"] = _iso(_now())
                self._write(upload_id, metadata)
            except UploadError:
                raise
            except OSError as exc:
                raise UploadStorageError("The server could not save this upload chunk.") from exc
            return self._public(metadata)

    def complete(
        self,
        upload_id: str,
        finalize: Optional[Callable[[Dict[str, Any], Path], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            metadata = self._read(upload_id)
            if metadata["status"] == "completed":
                return self._public(metadata)
            if metadata["status"] != "uploading":
                raise UploadConflict("Upload session cannot be completed.")
            if int(metadata["offset"]) != int(metadata["size"]):
                raise UploadConflict(f"Upload is incomplete at byte {metadata['offset']} of {metadata['size']}.")
            digest = hashlib.sha256()
            try:
                with self._payload_path(upload_id).open("rb") as payload:
                    for block in iter(lambda: payload.read(1024 * 1024), b""):
                        digest.update(block)
            except OSError as exc:
                raise UploadStorageError("The server could not verify this upload.") from exc
            actual = digest.hexdigest()
            if actual != str(metadata["sha256"]).lower():
                metadata["status"] = "failed"
                metadata["error"] = "The uploaded file failed SHA-256 verification."
                metadata["updated_at"] = _iso(_now())
                self._write(upload_id, metadata)
                raise UploadError(metadata["error"])
            result: Dict[str, Any] = {
                "filename": metadata["filename"],
                "size": metadata["size"],
                "sha256": actual,
                "path": str(self._payload_path(upload_id).resolve()),
            }
            if finalize:
                try:
                    callback_result = finalize(metadata, self._payload_path(upload_id)) or {}
                    result.update(callback_result)
                except UploadError:
                    raise
                except Exception as exc:
                    metadata["status"] = "failed"
                    metadata["error"] = str(exc)
                    metadata["updated_at"] = _iso(_now())
                    self._write(upload_id, metadata)
                    raise
            metadata["status"] = "completed"
            metadata["result"] = result
            metadata["updated_at"] = _iso(_now())
            self._write(upload_id, metadata)
            return self._public(metadata)

    def require_completed(self, upload_id: str, kind: str) -> Dict[str, Any]:
        with self._lock:
            metadata = self._read(upload_id)
            if metadata.get("kind") != kind or metadata.get("status") != "completed":
                raise UploadConflict("The upload is not ready for this operation.")
            return metadata

    def stage(self, upload_id: str, kind: str, destination: Path) -> Path:
        """Hard-link a completed prediction upload into a run, copying if needed."""
        if kind not in {"image", "video"}:
            raise UploadError("Only prediction media can be staged into a run.")
        with self._lock:
            metadata = self.require_completed(upload_id, kind)
            source = self._payload_path(upload_id)
            if not source.is_file():
                raise UploadNotFound("Completed upload data is no longer available.")
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / Path(metadata["filename"]).name
            suffix = 1
            while target.exists():
                target = destination / f"{Path(metadata['filename']).stem}-{suffix}{Path(metadata['filename']).suffix}"
                suffix += 1
            try:
                os.link(source, target)
            except OSError:
                try:
                    shutil.copyfile(source, target)
                except OSError as exc:
                    raise UploadStorageError("The server could not stage the completed upload.") from exc
            return target.resolve()

    def delete(self, upload_id: str) -> None:
        with self._lock:
            directory = self._session_dir(upload_id)
            if not directory.exists():
                return
            shutil.rmtree(directory)

    def cleanup_expired(self, now: Optional[datetime] = None) -> int:
        cutoff = (now or _now()) - timedelta(seconds=upload_retention_seconds())
        removed = 0
        with self._lock:
            if not self.root.is_dir():
                return 0
            for directory in self.root.iterdir():
                if not directory.is_dir() or directory.is_symlink():
                    continue
                try:
                    metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
                    updated = _parse_time(str(metadata.get("updated_at") or metadata.get("created_at")))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    updated = datetime.fromtimestamp(directory.stat().st_mtime, timezone.utc)
                if updated < cutoff:
                    shutil.rmtree(directory, ignore_errors=True)
                    removed += 1
        return removed

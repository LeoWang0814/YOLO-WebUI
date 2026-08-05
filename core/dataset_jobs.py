"""In-memory background jobs for responsive dataset preparation feedback."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from core.datasets import inspect_dataset


@dataclass
class DatasetPreparationJob:
    id: str
    path: str
    percent: int = 0
    message: str = "Queued"
    active: bool = True
    result: Optional[dict[str, Any]] = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {"id": self.id, "path": self.path, "percent": self.percent, "message": self.message, "active": self.active, "result": self.result}

    def update(self, percent: int, message: str) -> None:
        with self.lock:
            self.percent = percent
            self.message = message

    def finish(self, result: dict[str, Any]) -> None:
        with self.lock:
            self.percent = 100
            self.message = "Dataset ready" if result.get("status") == "ready" else "Dataset preparation blocked"
            self.result = result
            self.active = False


class DatasetPreparationManager:
    def __init__(self) -> None:
        self._jobs: dict[str, DatasetPreparationJob] = {}
        self._lock = threading.Lock()

    def start(self, path: str) -> DatasetPreparationJob:
        job = DatasetPreparationJob(id=uuid.uuid4().hex[:12], path=path)
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(target=self._run, args=(job,), name=f"dataset-{job.id}", daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Optional[DatasetPreparationJob]:
        with self._lock:
            return self._jobs.get(job_id)

    @staticmethod
    def _run(job: DatasetPreparationJob) -> None:
        try:
            result = inspect_dataset(job.path, progress=job.update)
        except Exception as exc:  # Defensive boundary so polling always reaches a terminal state.
            result = {"status": "blocked", "message": f"Dataset preparation failed: {exc}", "prepared_path": ""}
        job.finish(result)

"""Thread-safe orchestration for the single local Ultralytics process."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from core.gpu import get_system_status


Worker = Callable[["RunJob", "RunManager"], None]
ACTIVE_STAGES = {"queued", "resolving model", "preparing source", "running", "stopping"}


@dataclass
class RunJob:
    id: str
    kind: str
    run_dir: Path
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stage: str = "queued"
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    returncode: Optional[int] = None
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    stop_requested: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.stage in ACTIVE_STAGES


class RunConflictError(RuntimeError):
    pass


class RunManager:
    def __init__(self, max_log_lines: int = 2_000) -> None:
        self._lock = threading.RLock()
        self._jobs: Dict[str, RunJob] = {}
        self._active_id: Optional[str] = None
        self._max_log_lines = max_log_lines

    def start(self, kind: str, run_dir: Path, worker: Worker) -> RunJob:
        with self._lock:
            if self._active_id:
                current = self._jobs.get(self._active_id)
                if current and current.active:
                    raise RunConflictError(f"{current.kind.title()} is already running.")
            job = RunJob(id=uuid.uuid4().hex[:12], kind=kind, run_dir=run_dir)
            self._jobs[job.id] = job
            self._active_id = job.id
        thread = threading.Thread(target=self._run_worker, args=(job, worker), daemon=True, name=f"yolo-{kind}-{job.id}")
        thread.start()
        return job

    def _run_worker(self, job: RunJob, worker: Worker) -> None:
        try:
            worker(job, self)
            with self._lock:
                if job.stop_requested:
                    job.stage = "stopped"
                elif job.returncode not in (None, 0):
                    job.stage = "failed"
                    job.error = job.error or f"Process exited with code {job.returncode}."
                else:
                    job.stage = "completed"
        except Exception as exc:
            with self._lock:
                job.error = str(exc)
                job.stage = "stopped" if job.stop_requested else "failed"
            self.append_log(job, f"[error] {exc}")
        finally:
            with self._lock:
                job.process = None
                if self._active_id == job.id:
                    self._active_id = None

    def set_stage(self, job: RunJob, stage: str) -> None:
        with self._lock:
            job.stage = stage

    def update_details(self, job: RunJob, **details: Any) -> None:
        with self._lock:
            job.details.update(details)

    def append_log(self, job: RunJob, line: str) -> None:
        line = re.sub(r"\x1b\[[0-9;]*m", "", line).rstrip("\n")
        if not line:
            return
        with self._lock:
            job.logs.append(line)
            job.logs = job.logs[-self._max_log_lines :]
        job.run_dir.mkdir(parents=True, exist_ok=True)
        with (job.run_dir / "run.log").open("a", encoding="utf-8", errors="replace") as output:
            output.write(f"{line}\n")

    def run_command(self, job: RunJob, command: List[str], cwd: Path) -> int:
        self.set_stage(job, "running")
        self.append_log(job, "$ " + " ".join(command))
        execution_command = list(command)
        if execution_command and execution_command[0] == "yolo":
            bundled_cli = Path(sys.executable).with_name("yolo.exe" if os.name == "nt" else "yolo")
            if bundled_cli.is_file():
                execution_command[0] = str(bundled_cli)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        process = subprocess.Popen(
            execution_command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        with self._lock:
            job.process = process
        assert process.stdout is not None
        for line in process.stdout:
            self.append_log(job, line)
            if job.stop_requested and process.poll() is None:
                process.terminate()
        job.returncode = process.wait()
        return job.returncode

    def stop(self, job_id: Optional[str] = None) -> Optional[RunJob]:
        with self._lock:
            target_id = job_id or self._active_id
            job = self._jobs.get(target_id or "")
            if not job or not job.active:
                return None
            job.stop_requested = True
            job.stage = "stopping"
            process = job.process
        if process and process.poll() is None:
            process.terminate()
        self.append_log(job, "[status] Stop requested by user.")
        return job

    def get(self, job_id: str) -> Optional[RunJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def active_job(self) -> Optional[RunJob]:
        with self._lock:
            return self._jobs.get(self._active_id) if self._active_id else None

    def snapshot(self, job_id: str) -> Optional[Dict[str, object]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return {
                "id": job.id,
                "kind": job.kind,
                "run_dir": str(job.run_dir),
                "stage": job.stage,
                "active": job.active,
                "logs": list(job.logs),
                "error": job.error,
                "returncode": job.returncode,
                "details": dict(job.details),
                "created_at": job.created_at.isoformat(),
            }


def build_command(task: str, mode: str, args: Dict) -> Tuple[List[str], str]:
    command = ["yolo", task, mode]
    for key, value in args.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            value = str(value)
        elif isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value)
        command.append(f"{key}={value}")
    preview = " ".join(f'"{item}"' if " " in item else item for item in command)
    return command, preview


def write_run_metadata(run_dir: Path, args: Dict, command: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(args, indent=2, default=str), encoding="utf-8")
    (run_dir / "command.txt").write_text(command, encoding="utf-8")
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "system": get_system_status(),
    }
    (run_dir / "meta.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

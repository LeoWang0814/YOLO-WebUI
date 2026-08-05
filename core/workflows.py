"""Workflow helpers shared by the HTML workbench and background run manager."""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Tuple

import cv2
import pandas as pd
import plotly.graph_objects as go

from core.model_zoo import ensure_model, is_model_cached, model_choices
from core.runner import RunJob


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def model_catalog() -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    return model_choices()


def model_hint(choice: Optional[str]) -> str:
    choices, metadata = model_catalog()
    key = choice if choice in metadata else choices.get(choice or "")
    if not key or key not in metadata:
        return "Select a pretrained model to see download status."
    meta = metadata[key]
    path = ROOT / "weights" / meta["release"] / meta["filename"]
    if is_model_cached(key):
        return f"Cached at {path.relative_to(ROOT)}"
    size = meta.get("size_mb")
    return f"Will download {f'(~{size} MB)' if size else ''} when the run starts.".strip()


def resolve_model_path(
    source_kind: str,
    pretrained_choice: Optional[str],
    local_path: Optional[str],
    progress=None,
) -> Path:
    if source_kind == "pretrained":
        choices, metadata = model_catalog()
        key = pretrained_choice if pretrained_choice in metadata else choices.get(pretrained_choice or "")
        if not key:
            raise ValueError("Select a pretrained model.")
        return ensure_model(key, progress=progress)
    if not local_path:
        raise ValueError("Provide a local .pt model path or upload a model.")
    path = Path(local_path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if path.suffix.lower() != ".pt" or not path.is_file():
        raise ValueError("Local model must be an existing .pt file.")
    return path


def device_choices() -> List[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return [str(index) for index in range(torch.cuda.device_count())]
    except Exception:
        pass
    return []


def device_value(mode: str, single_gpu: Optional[str], multi_gpu: Iterable[str]) -> Optional[str]:
    if mode == "cpu":
        return "cpu"
    if mode == "single":
        if not single_gpu:
            raise ValueError("Select a GPU. No CUDA devices may be available.")
        return single_gpu
    if mode == "multi":
        values = [str(value) for value in multi_gpu if str(value).strip()]
        if not values:
            raise ValueError("Select at least one GPU.")
        return ",".join(values)
    if mode not in {"auto", ""}:
        raise ValueError("Invalid device selection.")
    return None


def allocate_run_dir(mode: str, requested_name: Optional[str] = None, create: bool = False) -> Path:
    requested_name = (requested_name or "").strip()
    root = RUNS_ROOT / mode
    if requested_name:
        if Path(requested_name).name != requested_name or requested_name in {".", ".."}:
            raise ValueError("Run name must be a single directory name.")
        run_dir = root / requested_name
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = root / f"{mode}-{stamp}"
        suffix = 1
        while run_dir.exists():
            run_dir = root / f"{mode}-{stamp}-{suffix}"
            suffix += 1
    if create:
        run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def discard_unstarted_run(run_dir: Path) -> None:
    """Remove a newly allocated run that never reached the manager."""
    candidate = run_dir.resolve()
    root = RUNS_ROOT.resolve()
    if candidate.parent.parent != root or candidate.parent.name not in {"train", "predict"}:
        raise ValueError("Invalid run cleanup path.")
    if candidate.is_dir():
        shutil.rmtree(candidate)


def resolve_run_path(relative_path: str) -> Path:
    candidate = (RUNS_ROOT / relative_path).resolve()
    root = RUNS_ROOT.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Invalid run path.")
    return candidate


def run_has_content(run_dir: Path) -> bool:
    return run_dir.exists() and any(run_dir.iterdir())


def save_uploaded_model(filename: str, stream: BinaryIO) -> Path:
    incoming = Path(filename or "").name
    if not incoming or Path(incoming).suffix.lower() != ".pt":
        raise ValueError("Only .pt model files are supported.")
    destination_dir = ROOT / "models"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / incoming
    suffix = 1
    while destination.exists():
        destination = destination_dir / f"{destination.stem}-{suffix}{destination.suffix}"
        suffix += 1
    with destination.open("wb") as target:
        shutil.copyfileobj(stream, target)
    return destination.resolve()


def stage_upload(filename: str, stream: BinaryIO, run_dir: Path) -> Path:
    incoming = Path(filename or "").name
    if not incoming:
        raise ValueError("Uploaded file has no filename.")
    source_dir = run_dir / ".source"
    source_dir.mkdir(parents=True, exist_ok=True)
    destination = source_dir / incoming
    suffix = 1
    while destination.exists():
        destination = source_dir / f"{destination.stem}-{suffix}{destination.suffix}"
        suffix += 1
    with destination.open("wb") as target:
        shutil.copyfileobj(stream, target)
    return destination.resolve()


def prepare_source(
    source_type: str,
    staged_images: List[Path],
    staged_video: Optional[Path],
    source_path: Optional[str],
) -> str:
    if source_type == "images":
        if not staged_images:
            raise ValueError("Upload at least one image.")
        return str(staged_images[0] if len(staged_images) == 1 else staged_images[0].parent)
    if source_type == "video":
        if not staged_video:
            raise ValueError("Upload a video.")
        return str(staged_video)
    if source_type == "path":
        if not source_path:
            raise ValueError("Provide a source path.")
        return source_path
    raise ValueError("Invalid source type.")


def _ffmpeg_path() -> Optional[str]:
    if ffmpeg := shutil.which("ffmpeg"):
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ensure_web_video(path: Path) -> Path:
    target = path.with_name(f"{path.stem}_web.mp4")
    try:
        if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
            return target
        if ffmpeg := _ffmpeg_path():
            result = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(path),
                    "-movflags",
                    "+faststart",
                    "-vf",
                    "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    str(target),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and target.exists():
                return target
        if path.suffix.lower() in {".mp4", ".webm"}:
            return path
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return path
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap.release()
            return path
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        while True:
            readable, frame = cap.read()
            if not readable:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        return target if target.exists() else path
    except Exception:
        return path


def collect_outputs(run_dir: Path, prepare_video: bool = True) -> Tuple[List[Path], Optional[Path]]:
    images = sorted(
        [
            path
            for path in run_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and ".source" not in path.relative_to(run_dir).parts
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    videos = sorted(
        [
            path
            for path in run_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES and ".source" not in path.relative_to(run_dir).parts
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    source_videos = [path for path in videos if not path.stem.endswith("_web")]
    video = (source_videos or videos)[0] if videos else None
    return images, ensure_web_video(video) if video and prepare_video else video


def find_results_csv(run_dir: Path) -> Optional[Path]:
    direct = run_dir / "results.csv"
    if direct.is_file():
        return direct
    candidates = list(run_dir.rglob("results.csv")) if run_dir.exists() else []
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def completed_epochs(run_dir: Path) -> int:
    csv_path = find_results_csv(run_dir)
    if not csv_path:
        return 0
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace") as source:
            next(source, None)
            return sum(1 for line in source if line.strip())
    except OSError:
        return 0


def _video_progress_details(path: Path) -> Dict[str, Any]:
    details: Dict[str, Any] = {"progress_kind": "video", "total": 0, "fps": 0.0}
    capture = cv2.VideoCapture(str(path))
    try:
        if capture.isOpened():
            details["total"] = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            details["fps"] = max(0.0, float(capture.get(cv2.CAP_PROP_FPS) or 0.0))
    finally:
        capture.release()
    return details


def prediction_progress_details(
    source_type: str,
    staged_images: List[Path],
    staged_video: Optional[Path],
    source_path: str = "",
) -> Dict[str, Any]:
    """Describe a prediction source before the worker starts producing log lines."""
    if source_type == "images":
        return {"progress_kind": "images", "total": len(staged_images), "source_label": "Uploaded images"}
    if source_type == "video" and staged_video:
        return {**_video_progress_details(staged_video), "source_label": staged_video.name}
    if source_type == "path":
        path = Path(source_path)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if path.is_dir():
            image_count = sum(1 for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
            if image_count:
                return {"progress_kind": "images", "total": image_count, "source_label": path.name or str(path)}
            videos = [item for item in path.rglob("*") if item.suffix.lower() in VIDEO_SUFFIXES]
            if len(videos) == 1:
                return {**_video_progress_details(videos[0]), "source_label": videos[0].name}
            return {"progress_kind": "media", "total": 0, "source_label": path.name or str(path)}
        if path.suffix.lower() in VIDEO_SUFFIXES and path.is_file():
            return {**_video_progress_details(path), "source_label": path.name}
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return {"progress_kind": "images", "total": 1, "source_label": path.name}
        return {"progress_kind": "media", "total": 0, "source_label": path.name or source_path}
    return {"progress_kind": "media", "total": 0, "source_label": "Prediction source"}


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def run_progress(job: Optional[RunJob] = None, operation: str = "predict") -> Dict[str, Any]:
    """Create a display-ready, task-specific progress snapshot."""
    if job is None:
        if operation == "train":
            return {
                "kind": "train",
                "title": "Training progress",
                "summary": "Waiting to start",
                "detail": "Epoch progress and the current training phase will appear here.",
                "percent": 0,
                "percent_label": "0%",
                "indeterminate": False,
                "stats": [("Measure", "Epochs"), ("Status", "Ready")],
            }
        return {
            "kind": "images",
            "title": "Prediction progress",
            "summary": "Waiting to start",
            "detail": "Processed images or video frames will be counted here.",
            "percent": 0,
            "percent_label": "0%",
            "indeterminate": False,
            "stats": [("Measure", "Inputs"), ("Status", "Ready")],
        }

    logs = "\n".join(job.logs)
    kind = "train" if job.kind == "train" else str(job.details.get("progress_kind") or "media")
    total = int(job.details.get("total") or 0)
    current = 0
    fps = float(job.details.get("fps") or 0.0)
    elapsed = _format_duration((datetime.now(timezone.utc) - job.created_at).total_seconds())

    if job.details.get("download_active"):
        download_percent = max(0, min(100, int(job.details.get("download_percent") or 0)))
        download_phase = str(job.details.get("download_phase") or "Downloading model...")
        download_indeterminate = bool(job.details.get("download_indeterminate"))
        return {
            "kind": kind,
            "title": "Downloading model",
            "summary": f"Model download {download_percent}%" if not download_indeterminate else "Preparing model download",
            "detail": download_phase,
            "percent": download_percent,
            "percent_label": f"{download_percent}%" if not download_indeterminate else "Live",
            "indeterminate": download_indeterminate,
            "stats": [("Phase", "Downloading model"), ("Elapsed", elapsed), ("Measure", "Model")],
        }

    if kind == "train":
        try:
            total = int(job.details.get("total") or run_metadata(job.run_dir).get("args", {}).get("epochs") or 0)
        except (TypeError, ValueError):
            total = 0
        current = completed_epochs(job.run_dir)
        if total:
            epoch_matches = [(int(left), int(right)) for left, right in re.findall(r"(?m)^\s*(\d+)\s*/\s*(\d+)\s+", logs)]
            current = max([current, *[left for left, right in epoch_matches if right == total]], default=current)
    elif kind == "video":
        frame_matches = [(int(left), int(right)) for left, right in re.findall(r"\bframe\s+(\d+)\s*/\s*(\d+)\b", logs, re.IGNORECASE)]
        if frame_matches:
            current, logged_total = frame_matches[-1]
            total = logged_total or total
    else:
        image_matches = [(int(left), int(right)) for left, right in re.findall(r"\bimage\s+(\d+)\s*/\s*(\d+)\b", logs, re.IGNORECASE)]
        if image_matches:
            current, logged_total = image_matches[-1]
            total = logged_total or total
            if kind in {"media", "stream"}:
                kind = "images"

    current = max(0, min(current, total)) if total else max(0, current)
    if job.stage == "completed" and total:
        current = total
    percent = int(round(current / total * 100)) if total else 0
    active_phase = {
        "queued": "Queued",
        "resolving model": "Resolving model",
        "preparing source": "Preparing source",
        "stopping": "Stopping process",
    }.get(job.stage)
    if job.stage == "running":
        if kind == "train":
            active_phase = "Training epochs" if "Starting training" in logs else "Preparing dataset"
        elif kind == "video":
            active_phase = "Processing video frames"
        elif kind == "images":
            active_phase = "Processing images"
        else:
            active_phase = "Processing source"
    phase = active_phase or job.stage.title()
    indeterminate = bool(job.active and (job.stage != "running" or not total or current == 0))

    if kind == "train":
        title = "Training progress"
        summary = f"Epoch {current} of {total}" if total else phase
        detail = f"{max(total - current, 0)} epochs remaining" if total and job.active else phase
        measure = "Epochs"
    elif kind == "video":
        title = "Video prediction"
        summary = f"Frame {current:,} of {total:,}" if total else (f"Frame {current:,}" if current else phase)
        if total and fps:
            detail = f"{_format_duration(current / fps)} of {_format_duration(total / fps)} video processed"
        elif total:
            detail = f"{max(total - current, 0):,} frames remaining" if job.active else phase
        else:
            detail = "Frame count will appear when the stream reports it." if job.active else phase
        measure = "Frames"
    elif kind == "images":
        title = "Image prediction"
        summary = f"{current:,} of {total:,} images" if total else (f"{current:,} images processed" if current else phase)
        detail = f"{max(total - current, 0):,} images remaining" if total and job.active else phase
        measure = "Images"
    else:
        title = "Prediction progress"
        summary = phase
        detail = "Progress updates as the source produces output."
        measure = "Items"

    return {
        "kind": kind,
        "title": title,
        "summary": summary,
        "detail": detail,
        "percent": percent,
        "percent_label": f"{percent}%" if total else "Live",
        "indeterminate": indeterminate,
        "stats": [("Phase", phase), ("Elapsed", elapsed), ("Measure", measure)],
    }


def _theme_layout(theme: str) -> Dict[str, Any]:
    dark = theme == "dark"
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#edf4f1" if dark else "#17221e"},
        "margin": {"l": 38, "r": 16, "t": 34, "b": 30},
        "legend": {"orientation": "h", "y": 1.16, "font": {"size": 11}},
        "xaxis": {"gridcolor": "#33413c" if dark else "#dce5e0", "zeroline": False},
        "yaxis": {"gridcolor": "#33413c" if dark else "#dce5e0", "zeroline": False},
    }


def _find_column(frame: pd.DataFrame, name: str) -> Optional[str]:
    if name in frame.columns:
        return name
    if name.endswith("(B)") and name[:-3] in frame.columns:
        return name[:-3]
    alternative = f"{name}(B)"
    return alternative if alternative in frame.columns else None


def _figure(frame: pd.DataFrame, series: List[Tuple[str, str]], theme: str) -> str:
    figure = go.Figure()
    x = frame["epoch"] if "epoch" in frame else list(range(1, len(frame) + 1))
    palette = ["#188cff", "#11a879", "#e25d4e", "#a56eff", "#d69b22", "#41a6a0"]
    for index, (candidate, label) in enumerate(series):
        column = _find_column(frame, candidate)
        if column:
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=pd.to_numeric(frame[column], errors="coerce"),
                    name=label,
                    mode="lines",
                    line={"width": 2, "color": palette[index % len(palette)]},
                )
            )
    if not figure.data:
        figure.add_annotation(text="No data yet", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    figure.update_layout(**_theme_layout(theme), hovermode="x unified")
    return figure.to_json()


def metrics_snapshot(run_dir: Path, theme: str = "light") -> Dict[str, Any]:
    csv_path = find_results_csv(run_dir)
    if not csv_path:
        return {"ready": False, "message": "Waiting for results.csv...", "figures": {}, "rows": [], "columns": [], "summary": []}
    try:
        frame = pd.read_csv(csv_path, encoding="utf-8", engine="python", skipinitialspace=True).tail(5000)
        frame.rename(columns=lambda name: str(name).strip().lstrip("\ufeff"), inplace=True)
    except Exception:
        return {"ready": False, "message": "Waiting for a complete results.csv...", "figures": {}, "rows": [], "columns": [], "summary": []}
    if frame.empty:
        return {"ready": False, "message": "Waiting for metrics...", "figures": {}, "rows": [], "columns": [], "summary": []}
    summary = []
    for label, key in (("mAP50-95", "metrics/mAP50-95(B)"), ("mAP50", "metrics/mAP50(B)"), ("Precision", "metrics/precision(B)"), ("Recall", "metrics/recall(B)"), ("Box loss", "train/box_loss")):
        column = _find_column(frame, key)
        value = pd.to_numeric(frame[column].iloc[-1], errors="coerce") if column else None
        summary.append({"label": label, "value": f"{float(value):.4f}" if value is not None and pd.notna(value) else "-"})
    rows = frame.tail(40).fillna("").to_dict(orient="records")
    return {
        "ready": True,
        "message": "Metrics update from results.csv",
        "summary": summary,
        "figures": {
            "loss": _figure(frame, [("train/box_loss", "Train box"), ("train/cls_loss", "Train class"), ("val/box_loss", "Val box"), ("val/cls_loss", "Val class")], theme),
            "quality": _figure(frame, [("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95"), ("metrics/precision(B)", "Precision"), ("metrics/recall(B)", "Recall")], theme),
            "learning_rate": _figure(frame, [("lr/pg0", "LR pg0"), ("lr/pg1", "LR pg1"), ("lr/pg2", "LR pg2")], theme),
        },
        "columns": list(frame.columns),
        "rows": rows,
    }


def run_metadata(run_dir: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"run_dir": run_dir, "kind": run_dir.parent.name, "name": run_dir.name, "command": "", "created": run_dir.stat().st_mtime}
    for filename, field in (("args.json", "args"), ("meta.json", "meta")):
        path = run_dir / filename
        if path.is_file():
            try:
                payload[field] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload[field] = {}
    command = run_dir / "command.txt"
    if command.is_file():
        payload["command"] = command.read_text(encoding="utf-8", errors="replace")
    return payload


def escaped_error(error: Exception) -> str:
    return html.escape(str(error))

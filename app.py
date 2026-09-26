"""FastAPI entrypoint for the self-hosted YOLOv10 workbench."""

from __future__ import annotations

import os
import json
import time
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from plotly.offline import get_plotlyjs
from starlette.concurrency import run_in_threadpool

from core.gpu import cuda_runtime_error, get_system_status
from core.dataset_jobs import DatasetPreparationManager
from core.datasets import FORMAT_CATALOG, prepared_dataset
from core.uploads import (
    UploadError,
    UploadStorageError,
    UploadStore,
    UploadTooLarge,
    max_upload_bytes,
    upload_chunk_bytes,
)
from core.runner import RunConflictError, RunJob, RunManager, build_command, write_run_metadata
from core.workflows import (
    ROOT,
    RUNS_ROOT,
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    allocate_run_dir,
    collect_outputs,
    discard_unstarted_run,
    device_choices,
    device_value,
    metrics_snapshot,
    model_catalog,
    model_hint,
    prediction_progress_details,
    prepare_source,
    resolve_model_path,
    resolve_run_path,
    run_has_content,
    run_progress,
    run_metadata,
    save_uploaded_dataset_archive,
    save_uploaded_model,
    stage_upload,
)
from web.docs import DOC_NAVIGATION, PRIMARY_CONTROLS, docs_page, docs_search_index, docs_slugs, parameter_docs
from web.forms import expert_groups, expert_values, form_control, form_list, integer, number, required_text


templates = Jinja2Templates(directory=str(ROOT / "templates"))
app = FastAPI(title="YOLOv10 Workbench", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
run_manager = RunManager()
dataset_manager = DatasetPreparationManager()
upload_store = UploadStore()
_last_upload_cleanup = 0.0
_upload_cleanup_stop = threading.Event()
_upload_cleanup_thread: Optional[threading.Thread] = None
ASSET_VERSION = str(max(
    (ROOT / "static" / "css" / "app.css").stat().st_mtime_ns,
    (ROOT / "static" / "js" / "app.js").stat().st_mtime_ns,
    (ROOT / "static" / "js" / "i18n.js").stat().st_mtime_ns,
    (ROOT / "static" / "js" / "sha256-worker.js").stat().st_mtime_ns,
))


def _template(request: Request, name: str, *, status_code: int = 200, headers: Optional[Dict[str, str]] = None, **context: Any):
    context.update({"request": request, "asset_version": ASSET_VERSION, "media_url": _media_url, "weight_path": _weight_path, "run_progress": run_progress})
    return templates.TemplateResponse(request=request, name=name, context=context, status_code=status_code, headers=headers)


def _theme(request: Request) -> str:
    return "dark" if request.headers.get("X-Theme") == "dark" else "light"


def _media_url(path: Path) -> str:
    return "/media/" + path.resolve().relative_to(RUNS_ROOT.resolve()).as_posix()


def _weight_path(job: RunJob, filename: str) -> Optional[str]:
    path = job.run_dir / "weights" / filename
    return str(path) if path.is_file() else None


def _model_choices() -> List[str]:
    choices, _ = model_catalog()
    return list(choices.keys())


def _defaults(operation: str) -> Dict[str, Any]:
    if operation == "train":
        return {
            "dataset_path": "",
            "pretrained_model": "yolov8n",
            "epochs": 100,
            "patience": 50,
            "imgsz": 640,
            "batch": -1,
            "workers": 8,
        }
    return {"pretrained_model": "yolov8n", "conf": 0.25, "iou": 0.7, "imgsz": 640}


def _page_context(request: Request, operation: str) -> Dict[str, Any]:
    values = _defaults(operation)
    model_choices = _model_choices()
    if values["pretrained_model"] not in model_choices and model_choices:
        values["pretrained_model"] = model_choices[0]
    active_job = run_manager.active_job()
    return {
        "current_page": operation,
        "operation": operation,
        "values": values,
        "model_choices": model_choices,
        "model_hint": model_hint(values["pretrained_model"]),
        "gpu_ids": device_choices(),
        "gpu_hint": cuda_runtime_error(),
        "expert_groups": expert_groups(operation),
        "form_control": form_control,
        "active_job": active_job,
        "dataset_formats": FORMAT_CATALOG,
    }


def _upload_from_form(form: Any, field_name: str, destination: Path) -> Optional[Path]:
    upload = form.get(field_name)
    if not getattr(upload, "filename", None):
        return None
    upload.file.seek(0)
    return stage_upload(upload.filename, upload.file, destination)


def _source_uploads(form: Any, run_dir: Path) -> tuple[List[Path], Optional[Path]]:
    images: List[Path] = []
    image_ids = [str(value).strip() for value in form.getlist("image_upload_id") if str(value).strip()]
    if image_ids:
        for upload_id in image_ids:
            images.append(upload_store.stage(upload_id, "image", run_dir / ".source"))
        return images, None
    for upload in form.getlist("images"):
        if getattr(upload, "filename", None):
            upload.file.seek(0)
            images.append(stage_upload(upload.filename, upload.file, run_dir))
    video_id = str(form.get("video_upload_id") or "").strip()
    if video_id:
        return images, upload_store.stage(video_id, "video", run_dir / ".source")
    video = _upload_from_form(form, "video", run_dir)
    return images, video


def _preview_model(form: Any) -> str:
    source_kind = str(form.get("model_source") or "pretrained")
    if source_kind == "local":
        return str(form.get("local_model") or "models/your-model.pt")
    selected = str(form.get("pretrained_model") or "")
    choices, metadata = model_catalog()
    key = selected if selected in metadata else choices.get(selected)
    if key and key in metadata:
        meta = metadata[key]
        return str(ROOT / "weights" / meta["release"] / meta["filename"])
    return "<select a pretrained model>"


def _validate_model_form(form: Any) -> None:
    source_kind = str(form.get("model_source") or "pretrained")
    if source_kind == "pretrained":
        selected = str(form.get("pretrained_model") or "")
        choices, metadata = model_catalog()
        if selected not in metadata and selected not in choices:
            raise ValueError("Select a pretrained model.")
        return
    if source_kind != "local":
        raise ValueError("Invalid model source.")
    local_model = str(form.get("local_model") or "").strip()
    model_upload_id = str(form.get("model_upload_id") or "").strip()
    if model_upload_id:
        upload_store.require_completed(model_upload_id, "model")
        return
    upload = form.get("model_upload")
    upload_name = str(getattr(upload, "filename", "") or "")
    if local_model:
        resolve_model_path("local", None, local_model)
    elif not upload_name:
        raise ValueError("Provide a local .pt model path or upload a model.")
    elif Path(upload_name).suffix.lower() != ".pt":
        raise ValueError("Only .pt model files are supported.")


def _validate_source_form(form: Any) -> None:
    source_type = str(form.get("source_type") or "images")
    if source_type == "images":
        image_uploads = [upload for upload in form.getlist("images") if getattr(upload, "filename", None)]
        if not form.getlist("image_upload_id") and not image_uploads:
            raise ValueError("Upload at least one image.")
        if any(Path(str(upload.filename)).suffix.lower() not in IMAGE_SUFFIXES for upload in image_uploads):
            raise ValueError("Only supported image files can be uploaded for prediction.")
        return
    if source_type == "video":
        video = form.get("video")
        if not str(form.get("video_upload_id") or "").strip() and not getattr(video, "filename", None):
            raise ValueError("Upload a video.")
        if getattr(video, "filename", None) and Path(str(video.filename)).suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError("Only supported video files can be uploaded for prediction.")
        return
    if source_type == "path":
        required_text(form, "source_path", "Source path")
        return
    raise ValueError("Invalid source type.")


def _build_args(form: Any, operation: str, run_dir: Path, model_path: str, source: Optional[str] = None, dataset_data: Optional[str] = None) -> Dict[str, Any]:
    args = expert_values(operation, form)
    device_mode = str(form.get("device_mode") or "auto")
    selected_multi_gpus = form_list(form, "multi_gpu")
    device = device_value(device_mode, form.get("single_gpu"), selected_multi_gpus)
    if operation == "train":
        batch = integer(form, "batch", "Batch", -1)
        if batch == 0:
            raise ValueError("Batch must be -1 for AutoBatch or a positive integer.")
        if batch == -1 and (
            device_mode == "cpu"
            or (device_mode == "multi" and len(selected_multi_gpus) > 1)
            or (device_mode in {"auto", ""} and not device_choices())
        ):
            raise ValueError("AutoBatch (-1) works only with one CUDA GPU. Enter a positive batch size for CPU or multiple GPUs.")
        if device_mode == "multi" and len(selected_multi_gpus) > 1 and args.get("rect"):
            raise ValueError("Rectangular training is not supported with multiple GPUs. Disable rect before starting the run.")
        args.update(
            {
                "data": dataset_data or str(form.get("data_path") or "<prepare a dataset folder>"),
                "model": model_path,
                "epochs": integer(form, "epochs", "Epochs", 1),
                "patience": integer(form, "patience", "Patience", 0),
                "imgsz": integer(form, "imgsz", "Image size", 32),
                "batch": batch,
                "workers": integer(form, "workers", "Workers", 0),
                "device": device,
                "project": str(run_dir.parent),
                "name": run_dir.name,
                "exist_ok": True,
                "verbose": True,
            }
        )
    else:
        args.update(
            {
                "model": model_path,
                "source": source or "<select a source>",
                "conf": number(form, "conf", "Confidence", 0.0, 0.25),
                "iou": number(form, "iou", "IoU", 0.0, 0.7),
                "imgsz": integer(form, "imgsz", "Image size", 32),
                "device": device,
                "project": str(run_dir.parent),
                "name": run_dir.name,
                "save": True,
                "exist_ok": True,
            }
        )
    return args


def _preview_source(form: Any) -> str:
    source_type = str(form.get("source_type") or "images")
    if source_type == "path":
        return str(form.get("source_path") or "<source path>")
    return "<uploaded video>" if source_type == "video" else "<uploaded image set>"


def _suggested_name(name: str) -> str:
    return f"{name}-new" if name else "experiment-new"


def _run_worker(job: RunJob, manager: RunManager, operation: str, args: Dict[str, Any], model_source: str, pretrained_model: str, local_model: str) -> None:
    manager.set_stage(job, "resolving model")
    manager.append_log(job, "[status] Resolving model...")
    manager.update_details(
        job,
        download_active=True,
        download_percent=0,
        download_phase="Checking model cache...",
        download_indeterminate=True,
    )

    def progress(_: float, desc: str = "") -> None:
        if manager.is_stop_requested(job):
            raise RuntimeError("Run was stopped by the user.")
        percent = max(0, min(100, int(round(_ * 100))))
        phase = desc or "Downloading model..."
        is_terminal = phase.lower().startswith(("verifying", "model cached"))
        manager.update_details(
            job,
            download_active=not is_terminal,
            download_percent=100 if is_terminal else percent,
            download_phase=phase,
            download_indeterminate=not is_terminal and percent <= 0,
        )
        if desc:
            manager.append_log(job, f"[download] {desc}")

    model_path = resolve_model_path(
        model_source,
        pretrained_model,
        local_model,
        progress=progress,
        cancelled=lambda: manager.is_stop_requested(job),
    )
    if manager.is_stop_requested(job):
        manager.append_log(job, "[status] Stopped before launching the Ultralytics process.")
        return
    manager.update_details(job, download_active=False, download_percent=100, download_phase="Model ready", download_indeterminate=False)
    args["model"] = str(model_path)
    command, preview = build_command("detect", operation, args)
    write_run_metadata(job.run_dir, args, preview)
    manager.set_stage(job, "preparing source")
    manager.append_log(job, f"[status] Using model: {model_path}")
    manager.append_log(job, "[status] Launching Ultralytics process...")
    manager.run_command(job, command, ROOT)


def _list_runs() -> List[Dict[str, Any]]:
    records = []
    for kind in ("train", "predict"):
        root = RUNS_ROOT / kind
        if not root.is_dir():
            continue
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            if not run_has_content(run_dir):
                continue
            images, video = collect_outputs(run_dir, prepare_video=False)
            records.append(
                {
                    "kind": kind,
                    "name": run_dir.name,
                    "updated": datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "artifacts": len(images) + int(video is not None),
                    "mtime": run_dir.stat().st_mtime,
                }
            )
    return sorted(records, key=lambda record: record["mtime"], reverse=True)


@app.get("/")
def workbench(request: Request, operation: str = "train"):
    if operation not in {"train", "predict"}:
        raise HTTPException(status_code=404, detail="Unknown operation")
    return _template(request, "workbench.html", **_page_context(request, operation))


@app.get("/runs")
def runs(request: Request):
    return _template(request, "runs.html", current_page="runs", runs=_list_runs())


DOC_PAGES = docs_slugs()


@app.get("/docs")
@app.get("/docs/{page}")
def docs(request: Request, page: str = "getting-started"):
    if page not in DOC_PAGES:
        raise HTTPException(status_code=404, detail="Documentation page not found")
    choices, metadata = model_catalog()
    return _template(
        request,
        "docs.html",
        current_page="docs",
        docs_page=docs_page(page),
        docs_navigation=DOC_NAVIGATION,
        docs_search_index=docs_search_index(),
        dataset_formats=FORMAT_CATALOG,
        primary_controls=PRIMARY_CONTROLS,
        train_parameters=parameter_docs("train"),
        predict_parameters=parameter_docs("predict"),
        model_choices=choices,
        model_metadata=metadata,
        image_suffixes=sorted(IMAGE_SUFFIXES),
        video_suffixes=sorted(VIDEO_SUFFIXES),
        launch_host=os.getenv("YOLOV10_WEBUI_HOST", "127.0.0.1"),
        launch_port=os.getenv("YOLOV10_WEBUI_PORT", "7860"),
    )


@app.get("/runs/{kind}/{name}")
def run_detail(request: Request, kind: str, name: str):
    if kind not in {"train", "predict"} or Path(name).name != name:
        raise HTTPException(status_code=404, detail="Run not found")
    run_dir = resolve_run_path(f"{kind}/{name}")
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    metadata = run_metadata(run_dir)
    images, video = collect_outputs(run_dir)
    log_path = run_dir / "run.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")[-20_000:] if log_path.is_file() else ""
    return _template(
        request,
        "run_detail.html",
        current_page="runs",
        run=metadata,
        images=images,
        video=video,
        metrics=metrics_snapshot(run_dir, _theme(request)),
        log_text=log_text,
    )


@app.get("/fragments/runtime")
def runtime_fragment(request: Request):
    return _template(request, "fragments/runtime.html", runtime=get_system_status())


@app.get("/fragments/model-hint")
def model_hint_fragment(request: Request, pretrained_model: str = ""):
    return _template(request, "fragments/model_hint.html", hint=model_hint(pretrained_model))


def _cleanup_upload_sessions() -> None:
    global _last_upload_cleanup
    now = time.monotonic()
    if now - _last_upload_cleanup < 3600:
        return
    _last_upload_cleanup = now
    upload_store.cleanup_expired()


def _upload_cleanup_loop() -> None:
    while not _upload_cleanup_stop.wait(3600):
        upload_store.cleanup_expired()


def start_upload_cleanup():
    global _upload_cleanup_thread
    _cleanup_upload_sessions()
    _upload_cleanup_stop.clear()
    if not _upload_cleanup_thread or not _upload_cleanup_thread.is_alive():
        _upload_cleanup_thread = threading.Thread(target=_upload_cleanup_loop, daemon=True, name="upload-cleanup")
        _upload_cleanup_thread.start()


def stop_upload_cleanup():
    _upload_cleanup_stop.set()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    start_upload_cleanup()
    try:
        yield
    finally:
        stop_upload_cleanup()


app.router.lifespan_context = _lifespan


def _upload_error_response(exc: Exception) -> JSONResponse:
    status = int(getattr(exc, "status_code", 422))
    if isinstance(exc, UploadStorageError) or isinstance(exc, OSError):
        status = 507
    return JSONResponse({"detail": str(exc)}, status_code=status)


def _finalize_resumable_upload(metadata: Dict[str, Any], payload: Path) -> Dict[str, Any]:
    kind = str(metadata.get("kind") or "")
    if kind == "dataset":
        with payload.open("rb") as stream:
            result = save_uploaded_dataset_archive(str(metadata["filename"]), stream)
        return {"dataset_path": result["dataset_path"], **result}
    if kind == "model":
        with payload.open("rb") as stream:
            model_path = save_uploaded_model(str(metadata["filename"]), stream)
        return {"path": str(model_path)}
    return {}


@app.post("/api/uploads")
async def create_upload_session(request: Request):
    """Create a persistent resumable upload session."""
    _cleanup_upload_sessions()
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise UploadError("Upload session body must be a JSON object.")
        result = await run_in_threadpool(
            upload_store.create,
            body.get("kind"),
            body.get("filename"),
            body.get("size"),
            body.get("sha256"),
        )
        result["chunk_size"] = upload_chunk_bytes()
        result["max_size"] = max_upload_bytes()
        return JSONResponse(result, status_code=201)
    except Exception as exc:
        return _upload_error_response(exc)


@app.get("/api/uploads/{upload_id}")
def get_upload_session(upload_id: str):
    _cleanup_upload_sessions()
    try:
        return JSONResponse(upload_store.get(upload_id))
    except Exception as exc:
        return _upload_error_response(exc)


@app.patch("/api/uploads/{upload_id}")
async def append_upload_chunk(request: Request, upload_id: str):
    _cleanup_upload_sessions()
    try:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > upload_chunk_bytes():
            raise UploadTooLarge("Upload chunk is larger than the configured chunk size.")
        offset = request.headers.get("upload-offset")
        if offset is None:
            raise UploadError("Upload-Offset header is required.")
        payload = await request.body()
        result = await run_in_threadpool(upload_store.append, upload_id, offset, payload)
        return JSONResponse(result)
    except (TypeError, ValueError) as exc:
        return _upload_error_response(exc)
    except Exception as exc:
        return _upload_error_response(exc)


@app.post("/api/uploads/{upload_id}/complete")
async def complete_upload_session(request: Request, upload_id: str):
    _cleanup_upload_sessions()
    try:
        # The checksum is part of session creation; accepting it here as an
        # optional assertion keeps the endpoint useful to non-browser clients.
        if request.headers.get("content-length", "0") not in {"", "0"}:
            body = await request.json()
            if body and body.get("sha256"):
                metadata = upload_store.get(upload_id)
                if str(body["sha256"]).lower() != str(metadata.get("sha256")).lower():
                    raise UploadError("Completion checksum does not match the upload session.")
        result = await run_in_threadpool(upload_store.complete, upload_id, _finalize_resumable_upload)
        return JSONResponse(result)
    except Exception as exc:
        return _upload_error_response(exc)


@app.delete("/api/uploads/{upload_id}")
def delete_upload_session(upload_id: str):
    _cleanup_upload_sessions()
    try:
        upload_store.delete(upload_id)
        return Response(status_code=204)
    except Exception as exc:
        return _upload_error_response(exc)


@app.post("/fragments/models/upload")
async def upload_model_fragment(request: Request):
    form = await request.form()
    upload = form.get("model_upload")
    if not getattr(upload, "filename", None):
        return _template(request, "fragments/upload_feedback.html", status_code=422, message="Choose a .pt file first.", success=False)
    try:
        upload.file.seek(0)
        model_path = save_uploaded_model(upload.filename, upload.file)
    except ValueError as exc:
        return _template(request, "fragments/upload_feedback.html", status_code=422, message=str(exc), success=False)
    return _template(request, "fragments/model_upload.html", model_path=model_path)


@app.post("/fragments/dataset/upload")
async def upload_dataset_fragment(request: Request):
    """Extract one browser-uploaded dataset archive before preparation."""
    archive: Any = None
    try:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_upload_bytes() + (1024 * 1024):
            raise UploadTooLarge("The uploaded dataset archive exceeds the configured upload size limit.")
        form = await request.form(max_files=1, max_fields=8)
        archive = form.get("dataset_archive")
        if not getattr(archive, "filename", None) or not hasattr(archive, "file"):
            raise ValueError("Drop one .zip dataset archive to upload.")
        result = await run_in_threadpool(save_uploaded_dataset_archive, archive.filename, archive.file)
    except UploadTooLarge as exc:
        return _upload_error_response(exc)
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    except OSError:
        return JSONResponse({"detail": "The server could not save this dataset upload."}, status_code=507)
    finally:
        close = getattr(archive, "close", None)
        if close:
            await close()
    return JSONResponse(result)


@app.post("/fragments/dataset/prepare")
@app.post("/fragments/validate-dataset")
async def dataset_fragment(request: Request):
    form = await request.form()
    path = str(form.get("dataset_path") or form.get("data_path") or "")
    options = {}
    raw_options = str(form.get("dataset_options") or "").strip()
    if raw_options:
        try:
            decoded = json.loads(raw_options)
            if isinstance(decoded, dict):
                options = decoded
        except json.JSONDecodeError:
            options = {}
    job = dataset_manager.start(path, options) if options else dataset_manager.start(path)
    return _template(request, "fragments/dataset_progress.html", job=job.snapshot())


@app.get("/fragments/dataset/prepare/{job_id}")
def dataset_progress_fragment(request: Request, job_id: str):
    job = dataset_manager.get(job_id)
    if not job:
        return _template(request, "fragments/dataset_preparation.html", status_code=404, dataset={"status": "blocked", "message": "Dataset preparation status is no longer available.", "prepared_path": ""})
    snapshot = job.snapshot()
    if snapshot["active"]:
        return _template(request, "fragments/dataset_progress.html", job=snapshot)
    return _template(request, "fragments/dataset_preparation.html", dataset=snapshot["result"])


@app.post("/fragments/preview/{operation}")
async def preview_fragment(request: Request, operation: str):
    if operation not in {"train", "predict"}:
        raise HTTPException(status_code=404, detail="Unknown operation")
    form = await request.form()
    try:
        run_dir = allocate_run_dir(operation, str(form.get("run_name") or ""), create=False)
        args = _build_args(form, operation, run_dir, _preview_model(form), _preview_source(form) if operation == "predict" else None)
        _, command = build_command("detect", operation, args)
    except ValueError as exc:
        command = f"Configuration error: {exc}"
    return _template(request, "fragments/command_preview.html", command=command)


@app.post("/runs/{operation}")
async def start_run(request: Request, operation: str):
    if operation not in {"train", "predict"}:
        raise HTTPException(status_code=404, detail="Unknown operation")
    form = await request.form()
    requested_name = str(form.get("run_name") or "").strip() if operation == "train" else ""
    run_dir: Optional[Path] = None
    run_dir_created = False
    try:
        proposed_dir = allocate_run_dir(operation, requested_name, create=False)
        if requested_name and proposed_dir.exists():
            return _template(request, "fragments/run_conflict.html", operation=operation, run_name=requested_name, suggestion=_suggested_name(requested_name))

        _validate_model_form(form)
        if operation == "predict":
            _validate_source_form(form)
        dataset_data = prepared_dataset(required_text(form, "dataset_path", "Dataset folder")) if operation == "train" else None
        args = _build_args(
            form,
            operation,
            proposed_dir,
            _preview_model(form),
            _preview_source(form) if operation == "predict" else None,
            dataset_data=dataset_data,
        )

        run_dir = proposed_dir
        run_dir.mkdir(parents=True, exist_ok=False)
        run_dir_created = True
        local_model = str(form.get("local_model") or "")
        model_upload_id = str(form.get("model_upload_id") or "").strip()
        if model_upload_id:
            model_upload = upload_store.require_completed(model_upload_id, "model")
            local_model = str(model_upload.get("result", {}).get("path") or "")
        model_upload = form.get("model_upload")
        if not local_model and getattr(model_upload, "filename", None):
            model_upload.file.seek(0)
            local_model = str(save_uploaded_model(model_upload.filename, model_upload.file))
        staged_images, staged_video = _source_uploads(form, run_dir)
        source = None
        progress_details: Dict[str, Any] = {"progress_kind": "train", "total": int(args.get("epochs") or 0)}
        if operation == "predict":
            source_type = str(form.get("source_type") or "images")
            source = prepare_source(
                source_type,
                staged_images,
                staged_video,
                str(form.get("source_path") or "").strip(),
            )
            args["source"] = source
            progress_details = prediction_progress_details(
                source_type,
                staged_images,
                staged_video,
                str(form.get("source_path") or "").strip(),
            )
        model_source = str(form.get("model_source") or "pretrained")
        pretrained_model = str(form.get("pretrained_model") or "")
        job = run_manager.start(
            operation,
            run_dir,
            lambda job, manager: _run_worker(job, manager, operation, args, model_source, pretrained_model, local_model),
        )
        job.details.update(progress_details)
    except RunConflictError as exc:
        if run_dir is not None and run_dir_created:
            discard_unstarted_run(run_dir)
        return _template(request, "fragments/run_error.html", status_code=409, operation=operation, title="Run slot unavailable", message=str(exc))
    except (ValueError, FileExistsError, OSError) as exc:
        if run_dir is not None and run_dir_created:
            discard_unstarted_run(run_dir)
        return _template(request, "fragments/run_error.html", status_code=422, operation=operation, title="Check configuration", message=str(exc))
    except Exception as exc:
        if run_dir is not None and run_dir_created:
            discard_unstarted_run(run_dir)
        return _template(request, "fragments/run_error.html", status_code=500, operation=operation, title="Unable to start run", message=str(exc))
    return _template(request, "fragments/start_response.html", job=job)


@app.post("/runs/{job_id}/stop")
def stop_run(request: Request, job_id: str):
    job = run_manager.get(job_id)
    if not job:
        return _template(request, "fragments/job_unavailable.html", status_code=404, message="This run is no longer available in memory.")
    if job.active:
        job = run_manager.stop(job_id) or job
    return _template(request, "fragments/stop_response.html", job=job)


@app.get("/fragments/jobs/{job_id}/inspector")
def inspector_fragment(request: Request, job_id: str):
    job = run_manager.get(job_id)
    if not job:
        return _template(request, "fragments/job_unavailable.html", status_code=404, message="Live status was lost after the service restarted.")
    return _template(request, "fragments/run_inspector.html", job=job)


@app.get("/fragments/jobs/{job_id}/activity")
def activity_fragment(request: Request, job_id: str):
    job = run_manager.get(job_id)
    if not job:
        return _template(
            request,
            "fragments/activity_unavailable.html",
            status_code=404,
            message="Live status was lost after the service restarted.",
        )
    return _template(request, "fragments/activity_response.html", job=job)


@app.get("/fragments/jobs/{job_id}/results")
def results_fragment(request: Request, job_id: str):
    job = run_manager.get(job_id)
    if not job:
        return _template(request, "fragments/results.html", job=None, run_dir=None, images=[], video=None, metrics={"ready": False})
    images, video = collect_outputs(job.run_dir)
    return _template(
        request,
        "fragments/results.html",
        job=job,
        run_dir=job.run_dir,
        images=images,
        video=video,
        metrics=metrics_snapshot(job.run_dir, _theme(request)),
    )


@app.get("/assets/plotly.min.js")
def plotly_asset():
    return Response(get_plotlyjs(), media_type="application/javascript", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/media/{relative_path:path}")
def media(relative_path: str):
    try:
        path = resolve_run_path(relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Media not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(path)


def launch() -> None:
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("YOLOV10_WEBUI_HOST", "127.0.0.1"),
        port=int(os.getenv("YOLOV10_WEBUI_PORT", "7860")),
        reload=False,
    )


if __name__ == "__main__":
    launch()

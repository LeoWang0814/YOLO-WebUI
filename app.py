"""FastAPI entrypoint for the self-hosted YOLOv10 workbench."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from plotly.offline import get_plotlyjs

from core.gpu import get_system_status
from core.dataset_jobs import DatasetPreparationManager
from core.datasets import FORMAT_CATALOG, prepared_dataset
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
    save_uploaded_model,
    stage_upload,
)
from web.docs import DOC_NAVIGATION, PRIMARY_CONTROLS, docs_page, docs_slugs, parameter_docs
from web.forms import expert_groups, expert_values, form_control, form_list, integer, number, required_text


templates = Jinja2Templates(directory=str(ROOT / "templates"))
app = FastAPI(title="YOLOv10 Workbench", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
run_manager = RunManager()
dataset_manager = DatasetPreparationManager()
ASSET_VERSION = str(max((ROOT / "static" / "css" / "app.css").stat().st_mtime_ns, (ROOT / "static" / "js" / "app.js").stat().st_mtime_ns))


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
            "batch": "auto",
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
    for upload in form.getlist("images"):
        if getattr(upload, "filename", None):
            upload.file.seek(0)
            images.append(stage_upload(upload.filename, upload.file, run_dir))
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
        if not any(getattr(upload, "filename", None) for upload in form.getlist("images")):
            raise ValueError("Upload at least one image.")
        return
    if source_type == "video":
        if not getattr(form.get("video"), "filename", None):
            raise ValueError("Upload a video.")
        return
    if source_type == "path":
        required_text(form, "source_path", "Source path")
        return
    raise ValueError("Invalid source type.")


def _build_args(form: Any, operation: str, run_dir: Path, model_path: str, source: Optional[str] = None, dataset_data: Optional[str] = None) -> Dict[str, Any]:
    args = expert_values(operation, form)
    device = device_value(str(form.get("device_mode") or "auto"), form.get("single_gpu"), form_list(form, "multi_gpu"))
    if operation == "train":
        batch_raw = str(form.get("batch") or "auto")
        batch = -1 if batch_raw == "auto" else integer(form, "batch", "Batch", 1)
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

    model_path = resolve_model_path(model_source, pretrained_model, local_model, progress=progress)
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


@app.post("/fragments/dataset/prepare")
@app.post("/fragments/validate-dataset")
async def dataset_fragment(request: Request):
    form = await request.form()
    path = str(form.get("dataset_path") or form.get("data_path") or "")
    job = dataset_manager.start(path)
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

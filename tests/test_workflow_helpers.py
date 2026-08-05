import threading
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import FormData

import app
from core import workflows
from core.runner import RunConflictError, RunJob, RunManager
from web.forms import expert_groups, expert_values
from web.docs import DOC_NAVIGATION, PARAMETER_OVERRIDES, docs_page, docs_search_index, docs_slugs, parameter_docs


def test_generated_run_dirs_are_unique(tmp_path, monkeypatch):
    monkeypatch.setattr(workflows, "RUNS_ROOT", tmp_path / "runs")

    first = workflows.allocate_run_dir("predict", create=True)
    second = workflows.allocate_run_dir("predict", create=True)

    assert first.exists()
    assert second.exists()
    assert first != second
    assert first.parent == tmp_path / "runs" / "predict"


def test_explicit_run_name_must_stay_inside_its_mode_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(workflows, "RUNS_ROOT", tmp_path / "runs")

    with pytest.raises(ValueError, match="single directory name"):
        workflows.allocate_run_dir("train", "../outside")


def test_collect_outputs_excludes_staged_source_files(tmp_path):
    run_dir = tmp_path / "run"
    source_dir = run_dir / ".source"
    source_dir.mkdir(parents=True)
    (source_dir / "original.jpg").write_bytes(b"source")
    processed = run_dir / "original_processed.jpg"
    processed.write_bytes(b"output")

    images, video = workflows.collect_outputs(run_dir)

    assert images == [processed]
    assert video is None


def test_uploaded_models_receive_collision_safe_names(tmp_path, monkeypatch):
    monkeypatch.setattr(workflows, "ROOT", tmp_path)
    source = tmp_path / "incoming.pt"
    source.write_bytes(b"weights")

    with source.open("rb") as stream:
        first = workflows.save_uploaded_model("incoming.pt", stream)
    with source.open("rb") as stream:
        second = workflows.save_uploaded_model("incoming.pt", stream)

    assert first.name == "incoming.pt"
    assert second.name == "incoming-1.pt"
    assert second.read_bytes() == b"weights"


@pytest.mark.parametrize("source_type", ["images", "video", "path"])
def test_prediction_source_modes_remain_supported(tmp_path, source_type):
    run_dir = tmp_path / "predict"
    run_dir.mkdir()
    image = run_dir / "image.jpg"
    video = run_dir / "video.mp4"
    image.write_bytes(b"image")
    video.write_bytes(b"video")

    source = workflows.prepare_source(
        source_type,
        [image] if source_type == "images" else [],
        video if source_type == "video" else None,
        "local/path" if source_type == "path" else "",
    )

    assert source


def test_expert_controls_exclude_overridden_fields_and_coerce_values():
    names = {key for _, fields in expert_groups("train") for key, _ in fields}
    assert "task" not in names
    assert "epochs" not in names
    values = expert_values("train", FormData([("expert__amp", "true"), ("expert__lr0", "0.02")]))
    assert values["amp"] is True
    assert values["lr0"] == 0.02


def test_metrics_snapshot_reports_epoch_metrics(tmp_path):
    results = pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "metrics/mAP50(B)": [0.1, 0.2, 0.3],
            "metrics/mAP50-95(B)": [0.05, 0.1, 0.15],
            "metrics/precision(B)": [0.2, 0.3, 0.4],
            "metrics/recall(B)": [0.3, 0.4, 0.5],
            "train/box_loss": [2.0, 1.0, 0.5],
        }
    )
    results.to_csv(tmp_path / "results.csv", index=False)

    snapshot = workflows.metrics_snapshot(tmp_path, "dark")

    assert snapshot["ready"] is True
    assert snapshot["summary"][0]["value"] == "0.1500"
    assert "loss" in snapshot["figures"]
    assert workflows.completed_epochs(tmp_path) == 3


def test_task_progress_uses_epoch_image_and_video_log_counters(tmp_path):
    training = RunJob(id="train", kind="train", run_dir=tmp_path / "train", stage="running", details={"total": 5})
    training.logs.extend(["Starting training for 5 epochs...", "        2/5         0G      1.234"])
    images = RunJob(id="images", kind="predict", run_dir=tmp_path / "images", stage="running", details={"progress_kind": "images", "total": 4})
    images.logs.append("image 3/4 C:/samples/cat.jpg: 640x480 1 cat, 12.0ms")
    video = RunJob(id="video", kind="predict", run_dir=tmp_path / "video", stage="running", details={"progress_kind": "video", "total": 120, "fps": 24.0})
    video.logs.append("video 1/1 (frame 48/120) C:/samples/clip.mp4: 640x480, 12.0ms")

    train_progress = workflows.run_progress(training)
    image_progress = workflows.run_progress(images)
    video_progress = workflows.run_progress(video)

    assert train_progress["summary"] == "Epoch 2 of 5"
    assert train_progress["percent"] == 40
    assert image_progress["summary"] == "3 of 4 images"
    assert image_progress["percent"] == 75
    assert video_progress["summary"] == "Frame 48 of 120"
    assert video_progress["detail"] == "0:02 of 0:05 video processed"


def test_task_progress_surfaces_model_download_in_same_card(tmp_path):
    job = RunJob(
        id="download",
        kind="predict",
        run_dir=tmp_path / "predict" / "download",
        stage="resolving model",
        details={
            "progress_kind": "images",
            "download_active": True,
            "download_percent": 42,
            "download_phase": "Downloading model... 42.0% 8.50 MB/s",
            "download_indeterminate": False,
        },
    )

    progress = workflows.run_progress(job)

    assert progress["title"] == "Downloading model"
    assert progress["summary"] == "Model download 42%"
    assert progress["percent"] == 42
    assert progress["stats"][-1] == ("Measure", "Model")
def test_run_manager_rejects_parallel_processes(tmp_path):
    manager = RunManager()
    release = threading.Event()

    def worker(job, manager):
        manager.set_stage(job, "running")
        release.wait(timeout=2)

    first = manager.start("predict", tmp_path / "first", worker)
    with pytest.raises(RunConflictError):
        manager.start("train", tmp_path / "second", worker)
    manager.stop(first.id)
    release.set()
    for _ in range(20):
        if not first.active:
            break
        time.sleep(0.02)
    assert first.stage == "stopped"


@pytest.fixture
def client():
    return TestClient(app.app)


class PassiveRunManager:
    def __init__(self, start_error=None, job=None):
        self.start_error = start_error
        self.job = job

    def active_job(self):
        return self.job if self.job and self.job.active else None

    def start(self, kind, run_dir, worker):
        if self.start_error:
            raise self.start_error
        self.job = RunJob(id="test-job", kind=kind, run_dir=run_dir)
        return self.job

    def get(self, job_id):
        return self.job if self.job and self.job.id == job_id else None

    def stop(self, job_id):
        if not self.job or self.job.id != job_id or not self.job.active:
            return None
        self.job.stage = "stopping"
        return self.job


def predict_form(**overrides):
    values = {
        "model_source": "pretrained",
        "pretrained_model": "yolov8n",
        "source_type": "images",
        "conf": "0.25",
        "iou": "0.7",
        "imgsz": "640",
        "device_mode": "cpu",
    }
    values.update(overrides)
    return values


def test_fastapi_renders_workbench_and_htmx_preview(client):
    page = client.get("/?operation=predict")
    assert page.status_code == 200
    assert "YOLOv10" in page.text
    assert 'name="video"' in page.text
    assert 'name="source_type" value="url"' not in page.text
    assert 'hx-include="#run-form"' in page.text
    assert 'hx-encoding="multipart/form-data"' in page.text
    assert 'data-start-run' in page.text
    assert 'id="app-feedback"' in page.text
    assert "/static/js/app.js?v=" in page.text
    assert page.text.index('class="configuration-form"') < page.text.index('class="execution-grid"')
    assert page.text.index('id="run-progress"') < page.text.index('id="run-log"') < page.text.index('id="run-results"')
    assert 'class="command-panel"' in page.text
    assert 'class="command-shell"' not in page.text
    assert 'class="theme-popover"' not in page.text
    assert 'id="media-viewer"' in page.text
    train = client.get("/?operation=train")
    assert 'name="dataset_path"' in train.text
    assert "/docs/datasets#supported-formats" in train.text
    assert 'class="dataset-doc-link" href="/docs/datasets#supported-formats" target="_blank" rel="noopener"' in train.text
    assert 'id="dataset-progress"' in train.text
    assert 'hx-indicator="#dataset-progress"' in train.text
    assert 'hx-trigger="input changed delay:700ms, blur"' not in train.text
    assert 'href="/?operation=train" aria-label="YOLOv10 Workbench home"' not in page.text


def test_docs_are_english_and_dataset_help_is_available(client):
    landing = client.get("/docs")
    datasets = client.get("/docs/datasets")

    assert landing.status_code == 200
    assert "Your first training run" in landing.text
    assert datasets.status_code == 200
    assert "Supported formats and conversion rules" in datasets.text
    assert "YOLOv8 Oriented Bounding Boxes" in datasets.text


def test_docs_routes_navigation_and_shared_reference_coverage(client):
    nav_slugs = {page["slug"] for section in DOC_NAVIGATION for page in section["pages"]}
    assert nav_slugs == docs_slugs()
    for slug in sorted(docs_slugs()):
        response = client.get(f"/docs/{slug}")
        assert response.status_code == 200
        assert "Documentation - YOLOv10 Workbench" in response.text
        assert 'data-doc-search' in response.text
        assert 'aria-current="page"' in response.text

    train_keys = {key for _, fields in expert_groups("train") for key, _ in fields}
    predict_keys = {key for _, fields in expert_groups("predict") for key, _ in fields}
    assert train_keys == {item["key"] for item in parameter_docs("train")}
    assert predict_keys == {item["key"] for item in parameter_docs("predict")}
    configuration = client.get("/docs/configuration")
    assert "Workbench-managed fields" in configuration.text
    assert "Advanced settings" in configuration.text


def test_docs_global_search_index_covers_each_page_and_section(client):
    index = docs_search_index()
    page_entries = [entry for entry in index if entry["kind"] == "Page"]
    section_entries = [entry for entry in index if entry["kind"] == "Section"]
    expected_sections = {
        f"{'/docs' if slug == 'getting-started' else f'/docs/{slug}'}#{anchor}"
        for slug in docs_slugs()
        for anchor, _ in docs_page(slug)["toc"]
    }

    assert {entry["url"] for entry in page_entries} == {
        "/docs" if slug == "getting-started" else f"/docs/{slug}"
        for slug in docs_slugs()
    }
    assert {entry["url"] for entry in section_entries} == expected_sections
    assert len(section_entries) == len(expected_sections)
    assert next(entry for entry in section_entries if entry["url"] == "/docs/datasets#cache")["title"] == "Cache and output"
    assert "checksum" in next(entry for entry in section_entries if entry["url"] == "/docs/models#download")["terms"]

    response = client.get("/docs/datasets")
    assert 'data-doc-search-index' in response.text
    assert 'role="combobox"' in response.text
    assert 'role="listbox"' in response.text


def test_every_visible_advanced_parameter_has_a_specific_reference_definition():
    for mode in ("train", "predict"):
        visible_keys = [key for _, fields in expert_groups(mode) for key, _ in fields]
        documented = parameter_docs(mode)

        assert [item["key"] for item in documented] == visible_keys
        assert all(item["description"] == PARAMETER_OVERRIDES[item["key"]] for item in documented)
        assert all(
            item["description"] != f"Ultralytics {item['key'].replace('_', ' ')} setting for {mode} mode."
            for item in documented
        )


def test_dataset_preparation_hides_internal_detection_evidence(monkeypatch, client):
    class CompletedJob:
        def snapshot(self):
            return {"id": "completed", "active": False, "result": {"status": "ready", "format": "coco", "images": 2, "objects": 3, "classes": ["cat"], "splits": {"train": 1, "val": 1}, "prepared_path": "C:/managed/data.yaml", "evidence": "annotations.coco.json: internal detection evidence"}}

    class CompletedManager:
        def start(self, _):
            return CompletedJob()

        def get(self, _):
            return CompletedJob()

    monkeypatch.setattr(app, "dataset_manager", CompletedManager())

    response = client.post("/fragments/dataset/prepare", data={"dataset_path": "C:/sample"})
    completed = client.get("/fragments/dataset/prepare/completed")

    assert response.status_code == 200
    assert "Preparing a local YOLOv10 Detect dataset" in response.text
    assert 'data-dataset-job="completed"' in response.text
    assert completed.status_code == 200
    assert "View dataset summary" in completed.text
    assert "annotations.coco.json" not in completed.text
    assert 'href="/docs/datasets#supported-formats" target="_blank" rel="noopener"' in completed.text

    preview = client.post(
        "/fragments/preview/predict",
        data={"model_source": "pretrained", "pretrained_model": "yolov8n", "source_type": "path", "source_path": "example-images/cat01.jpg", "conf": "0.25", "iou": "0.7", "imgsz": "640", "device_mode": "cpu"},
        headers={"HX-Request": "true"},
    )
    assert preview.status_code == 200
    assert "yolo detect predict" in preview.text
    assert "exist_ok=True" in preview.text


def test_prediction_image_upload_reaches_staging_directory(tmp_path, monkeypatch, client):
    runs_root = tmp_path / "runs"
    manager = PassiveRunManager()
    monkeypatch.setattr(workflows, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(app, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(app, "run_manager", manager)

    response = client.post(
        "/runs/predict",
        data=predict_form(),
        files=[("images", ("cat.jpg", b"image", "image/jpeg"))],
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    assert manager.job is not None
    assert (manager.job.run_dir / ".source" / "cat.jpg").read_bytes() == b"image"
    assert manager.job.details["progress_kind"] == "images"
    assert manager.job.details["total"] == 1


def test_activity_refreshes_log_with_oob_status_and_progress(tmp_path, monkeypatch, client):
    job = RunJob(
        id="live-job",
        kind="predict",
        run_dir=tmp_path / "predict" / "live-job",
        stage="running",
        logs=["image 1/2 sample.jpg: 640x480"],
        details={"progress_kind": "images", "total": 2},
    )
    monkeypatch.setattr(app, "run_manager", PassiveRunManager(job=job))

    response = client.get("/fragments/jobs/live-job/activity", headers={"HX-Request": "true"})

    assert response.status_code == 200
    assert response.text.index('id="run-log"') < response.text.index('id="run-inspector"')
    assert response.text.count('hx-swap-oob="outerHTML"') == 2
    assert "1 of 2 images" in response.text


def test_completed_run_inspector_has_open_and_new_actions(tmp_path, monkeypatch, client):
    job = RunJob(
        id="done",
        kind="predict",
        run_dir=tmp_path / "predict" / "done",
        stage="completed",
        returncode=0,
    )
    monkeypatch.setattr(app, "run_manager", PassiveRunManager(job=job))

    response = client.get("/fragments/jobs/done/inspector", headers={"HX-Request": "true"})

    assert response.status_code == 200
    assert "Open run" in response.text
    assert "New prediction" in response.text


def test_start_validation_is_visible_and_does_not_create_run(tmp_path, monkeypatch, client):
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(workflows, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(app, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(app, "run_manager", PassiveRunManager())

    response = client.post("/runs/predict", data=predict_form(), headers={"HX-Request": "true"})

    assert response.status_code == 422
    assert "Upload at least one image" in response.text
    assert "Retry" in response.text
    assert not (runs_root / "predict").exists()


def test_explicit_gpu_without_device_has_visible_error(tmp_path, monkeypatch, client):
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(workflows, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(app, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(app, "run_manager", PassiveRunManager())

    response = client.post(
        "/runs/predict",
        data=predict_form(source_type="path", source_path="example-images/cat01.jpg", device_mode="single"),
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 422
    assert "Select a GPU" in response.text
    assert not (runs_root / "predict").exists()


def test_run_conflict_removes_unstarted_directory(tmp_path, monkeypatch, client):
    runs_root = tmp_path / "runs"
    manager = PassiveRunManager(start_error=RunConflictError("Train is already running."))
    monkeypatch.setattr(workflows, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(app, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(app, "run_manager", manager)

    response = client.post(
        "/runs/predict",
        data=predict_form(source_type="path", source_path="example-images/cat01.jpg"),
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 409
    assert "Train is already running" in response.text
    assert not list((runs_root / "predict").glob("*"))


def test_invalid_model_upload_returns_targeted_feedback(client):
    response = client.post(
        "/fragments/models/upload",
        files={"model_upload": ("weights.txt", b"invalid", "text/plain")},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 422
    assert 'id="model-upload-status"' in response.text
    assert "Only .pt model files" in response.text


def test_stale_job_requests_return_visible_terminal_fragments(monkeypatch, client):
    monkeypatch.setattr(app, "run_manager", PassiveRunManager())

    inspector = client.get("/fragments/jobs/missing/inspector", headers={"HX-Request": "true"})
    results = client.get("/fragments/jobs/missing/results", headers={"HX-Request": "true"})

    assert inspector.status_code == 404
    assert "Run disconnected" in inspector.text
    assert "hx-trigger" not in inspector.text
    assert results.status_code == 200
    assert "Results will appear" in results.text


def test_stop_after_completion_returns_current_state(tmp_path, monkeypatch, client):
    job = RunJob(id="finished", kind="predict", run_dir=tmp_path / "predict" / "finished", stage="completed", returncode=0)
    monkeypatch.setattr(app, "run_manager", PassiveRunManager(job=job))

    response = client.post("/runs/finished/stop", headers={"HX-Request": "true"})

    assert response.status_code == 200
    assert "Completed" in response.text
    assert "Open run" in response.text


def test_media_route_rejects_paths_outside_runs(tmp_path, monkeypatch, client):
    monkeypatch.setattr(app, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(workflows, "RUNS_ROOT", tmp_path / "runs")
    allowed = tmp_path / "runs" / "predict" / "demo" / "result.jpg"
    allowed.parent.mkdir(parents=True)
    allowed.write_bytes(b"result")

    assert client.get("/media/predict/demo/result.jpg").status_code == 200
    assert client.get("/media/../../outside.txt").status_code == 404

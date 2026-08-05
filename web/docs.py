"""Structured product documentation metadata for the local Workbench."""

from __future__ import annotations

from typing import Any, Dict, List

from core.args_schema import build_grouped_defaults
from web.forms import MANAGED_FIELDS


DOC_NAVIGATION = [
    {"label": "START HERE", "pages": [
        {"slug": "getting-started", "title": "Getting started", "icon": "rocket"},
        {"slug": "overview", "title": "Product overview", "icon": "layout-dashboard"},
    ]},
    {"label": "WORKFLOWS", "pages": [
        {"slug": "train", "title": "Train", "icon": "graduation-cap"},
        {"slug": "datasets", "title": "Dataset preparation", "icon": "folder-search-2"},
        {"slug": "models", "title": "Models and weights", "icon": "boxes"},
        {"slug": "predict", "title": "Predict", "icon": "scan-line"},
        {"slug": "runs", "title": "Runs and results", "icon": "history"},
    ]},
    {"label": "REFERENCE", "pages": [
        {"slug": "configuration", "title": "Configuration reference", "icon": "sliders-horizontal"},
        {"slug": "runtime", "title": "Runtime and hardware", "icon": "cpu"},
        {"slug": "storage-and-privacy", "title": "Files, storage and privacy", "icon": "hard-drive"},
        {"slug": "troubleshooting", "title": "Troubleshooting", "icon": "life-buoy"},
        {"slug": "glossary", "title": "Glossary and limits", "icon": "book-a"},
    ]},
]


PAGE_META: Dict[str, Dict[str, Any]] = {
    "getting-started": {"eyebrow": "YOLOV10 WORKBENCH", "title": "Get started", "summary": "A local, repeatable workflow for YOLO Detect training and prediction.", "toc": [("requirements", "Requirements"), ("launch", "Launch the Workbench"), ("first-training", "First training run"), ("first-prediction", "First prediction")], "related": ["datasets", "train", "predict"]},
    "overview": {"eyebrow": "PRODUCT OVERVIEW", "title": "A local workflow for YOLO Detect", "summary": "Understand the Workbench boundary, the pages it provides, and the lifecycle of a local run.", "toc": [("what-it-is", "What it is"), ("workflow", "Workflow"), ("limits", "Product limits")], "related": ["getting-started", "runtime", "storage-and-privacy"]},
    "train": {"eyebrow": "WORKFLOW", "title": "Train", "summary": "Create, monitor, stop, and review an axis-aligned detection training run.", "toc": [("before-starting", "Before starting"), ("train-form", "Configuration form"), ("execution", "Execution and progress"), ("outputs", "Training outputs")], "related": ["datasets", "models", "configuration", "runs"]},
    "datasets": {"eyebrow": "TRAIN · DATASET", "title": "Dataset preparation", "summary": "Inspect one local folder, prepare a strict YOLO Detect dataset, and keep the source unchanged.", "toc": [("from-zero", "From zero to Ready"), ("progress", "Preparation progress"), ("supported-formats", "Supported formats"), ("lossless", "Strict lossless policy"), ("cache", "Cache and output"), ("dataset-blockers", "Common blockers")], "related": ["train", "configuration", "troubleshooting"]},
    "models": {"eyebrow": "WORKFLOW", "title": "Models and weights", "summary": "Choose verified pretrained weights or a local checkpoint without losing track of where files are stored.", "toc": [("pretrained", "Pretrained models"), ("local-models", "Local models and uploads"), ("download", "Downloads and verification"), ("errors", "Model errors")], "related": ["train", "predict", "runtime"]},
    "predict": {"eyebrow": "WORKFLOW", "title": "Predict", "summary": "Run a chosen model against local images, video, or a filesystem path and inspect saved outputs.", "toc": [("sources", "Choose a source"), ("prediction-settings", "Prediction settings"), ("run-prediction", "Run and inspect"), ("viewer", "Media viewer")], "related": ["models", "runs", "configuration"]},
    "runs": {"eyebrow": "WORKFLOW", "title": "Runs and results", "summary": "Find completed work, read its command and logs, and inspect metrics or generated media.", "toc": [("run-list", "Run list"), ("run-details", "Run details"), ("states", "Run states"), ("artifacts", "Artifacts")], "related": ["train", "predict", "storage-and-privacy"]},
    "configuration": {"eyebrow": "REFERENCE", "title": "Configuration reference", "summary": "Primary controls and every Advanced settings field currently exposed by the Workbench.", "toc": [("primary-controls", "Primary controls"), ("train-advanced", "Train advanced settings"), ("predict-advanced", "Predict advanced settings"), ("managed", "Workbench-managed fields")], "related": ["train", "predict", "runtime"]},
    "runtime": {"eyebrow": "REFERENCE", "title": "Runtime and hardware", "summary": "How local execution, device selection, downloads, and the single-run queue behave.", "toc": [("device-selection", "Device selection"), ("performance", "Performance controls"), ("queue", "Run queue"), ("download-progress", "Model downloads")], "related": ["configuration", "models", "troubleshooting"]},
    "storage-and-privacy": {"eyebrow": "REFERENCE", "title": "Files, storage and privacy", "summary": "Where the Workbench stores data and what stays on your machine.", "toc": [("locations", "Storage locations"), ("data-lifecycle", "Data lifecycle"), ("privacy", "Privacy and access"), ("retention", "Retention")], "related": ["datasets", "runs", "troubleshooting"]},
    "troubleshooting": {"eyebrow": "REFERENCE", "title": "Troubleshooting", "summary": "Resolve common dataset, model, source, runtime, and output failures with the evidence shown in the UI.", "toc": [("dataset-errors", "Dataset errors"), ("model-errors", "Model errors"), ("run-errors", "Run errors"), ("output-errors", "Output errors")], "related": ["datasets", "models", "runtime"]},
    "glossary": {"eyebrow": "REFERENCE", "title": "Glossary and limits", "summary": "Shared vocabulary and the current product boundaries for this local YOLO Detect Workbench.", "toc": [("glossary", "Glossary"), ("supported", "Supported capabilities"), ("not-supported", "Not supported")], "related": ["overview", "datasets", "troubleshooting"]},
}


PRIMARY_CONTROLS = [
    ("Train", "Dataset folder", "Local directory inspected into a prepared YOLO Detect dataset before a run can start."),
    ("Train", "Model", "Select a verified pretrained model, a local .pt path, or upload a .pt file."),
    ("Train", "Epochs", "Maximum number of training epochs."),
    ("Train", "Patience", "Epochs without improvement allowed before early stopping; 0 disables patience-based stopping."),
    ("Train", "Image size", "Training image size; larger values require more compute and memory."),
    ("Train", "Batch", "Images per optimization step; Auto delegates batch sizing to Ultralytics."),
    ("Train", "Workers", "Data-loader worker processes; lower this if the machine is resource constrained."),
    ("Train / Predict", "Device", "Auto, CPU, one CUDA GPU, or multiple CUDA GPUs when available."),
    ("Predict", "Source", "Uploaded images, one uploaded video, or a local filesystem path. URLs are not supported."),
    ("Predict", "Confidence", "Minimum detection confidence retained in output."),
    ("Predict", "IoU", "Overlap threshold used by non-maximum suppression."),
]


PARAMETER_OVERRIDES = {
    "amp": "Use automatic mixed precision when supported by the selected device.",
    "cache": "Cache training images in memory or on disk when memory and storage allow.",
    "close_mosaic": "Disable mosaic augmentation for the final training epochs.",
    "cos_lr": "Use cosine learning-rate scheduling.",
    "deterministic": "Request deterministic behavior where the underlying stack supports it.",
    "fraction": "Use a fraction of the dataset for a faster experiment.",
    "lr0": "Initial learning rate.",
    "lrf": "Final learning-rate multiplier.",
    "mosaic": "Probability of mosaic augmentation.",
    "optimizer": "Optimizer selection; Auto lets Ultralytics choose.",
    "resume": "Resume an interrupted compatible training run.",
    "save_period": "Save checkpoints every specified number of epochs; -1 keeps the default behavior.",
    "seed": "Random seed for repeatable data order and augmentation where supported.",
    "val": "Run validation during training.",
    "vid_stride": "Process every Nth video frame during prediction.",
    "stream_buffer": "Buffer frames for streaming sources instead of dropping late frames.",
    "agnostic_nms": "Apply non-maximum suppression across classes.",
    "classes": "Limit prediction to selected class IDs.",
    "save_txt": "Save predictions as text labels alongside visual output.",
    "save_conf": "Include confidence values in saved text labels.",
    "save_crop": "Save cropped detections as additional artifacts.",
}


def parameter_docs(mode: str) -> List[Dict[str, Any]]:
    """Return one documented row for every advanced field rendered in the UI."""
    rows: List[Dict[str, Any]] = []
    for group, values in build_grouped_defaults(mode).items():
        for key, default in values.items():
            if key in MANAGED_FIELDS[mode]:
                continue
            rows.append({
                "group": group,
                "key": key,
                "default": repr(default),
                "type": type(default).__name__,
                "description": PARAMETER_OVERRIDES.get(key, f"Ultralytics {key.replace('_', ' ')} setting for {mode} mode."),
            })
    return rows


def docs_page(slug: str) -> Dict[str, Any]:
    return {"slug": slug, **PAGE_META[slug]}


def docs_slugs() -> set[str]:
    return set(PAGE_META)

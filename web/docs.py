"""Structured product documentation metadata for the local Workbench."""

from __future__ import annotations

from typing import Any, Dict, List

from web.forms import expert_groups


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


# Search aliases deliberately live beside the page map so a new Docs section has a
# clear, maintained place to define the vocabulary users are likely to search for.
DOC_SECTION_KEYWORDS: Dict[tuple[str, str], List[str]] = {
    ("getting-started", "requirements"): ["installation", "python", "requirements", "local", "self-hosted"],
    ("getting-started", "launch"): ["start", "command", "uvicorn", "host", "port", "environment variables"],
    ("getting-started", "first-training"): ["dataset", "inspect", "ready", "epochs", "weights"],
    ("getting-started", "first-prediction"): ["images", "video", "path", "confidence", "iou"],
    ("overview", "what-it-is"): ["local", "ultralytics", "cli", "detect", "workflow"],
    ("overview", "workflow"): ["command", "model", "artifacts", "logs", "lifecycle"],
    ("overview", "limits"): ["unsupported", "obb", "segmentation", "url", "concurrent"],
    ("train", "before-starting"): ["dataset", "ready", "model", "single run"],
    ("train", "train-form"): ["epochs", "batch", "workers", "device", "advanced settings"],
    ("train", "execution"): ["progress", "stop", "logs", "download", "queue"],
    ("train", "outputs"): ["weights", "metrics", "results", "run directory"],
    ("datasets", "from-zero"): ["folder", "inspect", "convert", "yolo", "coco", "voc"],
    ("datasets", "progress"): ["checking", "verifying", "preparing", "percentage", "inspection"],
    ("datasets", "supported-formats"): ["coco", "voc", "createml", "csv", "yolo", "format conversion"],
    ("datasets", "lossless"): ["obb", "segmentation", "classification", "bounding boxes", "incompatible"],
    ("datasets", "cache"): ["prepared dataset", "fingerprint", "manifest", "data.yaml", "labels"],
    ("datasets", "dataset-blockers"): ["ambiguous", "missing image", "invalid labels", "splits", "errors"],
    ("models", "pretrained"): ["catalog", "weights", "cache", "release", "model"],
    ("models", "local-models"): ["upload", "checkpoint", ".pt", "filename", "path"],
    ("models", "download"): ["checksum", "sha256", "retry", "partial download", "verification"],
    ("models", "errors"): ["download failed", "checksum mismatch", "invalid model", "retry"],
    ("predict", "sources"): ["images", "video", "local path", "uploads", "url unsupported"],
    ("predict", "prediction-settings"): ["confidence", "iou", "image size", "device", "nms"],
    ("predict", "run-prediction"): ["run", "progress", "output", "artifacts"],
    ("predict", "viewer"): ["zoom", "pan", "open original", "mp4", "media"],
    ("runs", "run-list"): ["search", "filter", "train", "predict", "updated"],
    ("runs", "run-details"): ["command", "logs", "metadata", "metrics", "results"],
    ("runs", "states"): ["queued", "running", "completed", "failed", "stopped", "disconnected"],
    ("runs", "artifacts"): ["args.json", "command.txt", "run.log", "weights", "media"],
    ("configuration", "primary-controls"): ["epochs", "batch", "workers", "device", "confidence", "iou"],
    ("configuration", "train-advanced"): ["ultralytics", "optimizer", "augmentation", "hyperparameters"],
    ("configuration", "predict-advanced"): ["output", "video stride", "class filter", "nms"],
    ("configuration", "managed"): ["task", "mode", "data", "source", "model", "project"],
    ("runtime", "device-selection"): ["cpu", "cuda", "gpu", "auto", "multiple gpus"],
    ("runtime", "performance"): ["batch", "workers", "amp", "cache", "memory", "throughput"],
    ("runtime", "queue"): ["single active run", "conflict", "concurrent", "training", "prediction"],
    ("runtime", "download-progress"): ["model download", "percentage", "checksum", "verified weights"],
    ("storage-and-privacy", "locations"): ["runs", "weights", "models", "prepared datasets", "uploads"],
    ("storage-and-privacy", "data-lifecycle"): ["source", "immutable", "staged", "cache", "artifacts"],
    ("storage-and-privacy", "privacy"): ["local-only", "network", "authentication", "remote urls"],
    ("storage-and-privacy", "retention"): ["cleanup", "delete", "persist", "storage", "cache"],
    ("troubleshooting", "dataset-errors"): ["blocked", "ambiguous", "missing image", "invalid box", "empty split"],
    ("troubleshooting", "model-errors"): ["download", "checksum", "upload", ".pt", "path"],
    ("troubleshooting", "run-errors"): ["gpu", "cuda", "run name", "failed", "stopped", "disconnected"],
    ("troubleshooting", "output-errors"): ["artifacts", "video", "media playback", "logs", "results"],
    ("glossary", "glossary"): ["artifact", "cache", "detect", "obb", "split", "source"],
    ("glossary", "supported"): ["training", "prediction", "dataset", "local", "models"],
    ("glossary", "not-supported"): ["urls", "obb", "segmentation", "pose", "classification", "cloud"],
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
    "agnostic_nms": "Run non-maximum suppression without respecting class IDs; overlapping boxes from different classes can suppress one another.",
    "amp": "Use Automatic Mixed Precision during training. It can reduce CUDA memory use and improve speed; this Workbench defaults it to false, while the upstream Ultralytics default is true.",
    "augment": "Use test-time augmentation for prediction. It performs extra inference passes and can improve recall at a substantial speed cost.",
    "auto_augment": "Classification-only automatic augmentation policy (randaugment, autoaugment, or augmix). It does not affect Detect training.",
    "batch": "Number of images processed per optimization step. In prediction, it is the inference batch size; the primary Train control owns the training value.",
    "bgr": "Probability of swapping image channels from RGB to BGR as a detection augmentation.",
    "box": "Gain applied to the bounding-box regression loss during training.",
    "cache": "Training data cache mode: false disables caching, true or ram caches in memory, and disk caches on disk. It is a training-only control.",
    "cfg": "Optional path to an Ultralytics configuration file that overrides defaults. Leave it empty in the Workbench unless you intentionally maintain such a file.",
    "classes": "Optional class ID or list of class IDs to retain in prediction results, for example 0 or [0, 2, 3].",
    "close_mosaic": "Number of final training epochs for which mosaic augmentation is disabled. Set 0 to keep mosaic enabled throughout training.",
    "cls": "Gain applied to the classification loss during training; it scales with image size in Ultralytics Detect training.",
    "conf": "Object-confidence threshold. It defaults to 0.25 for prediction and 0.001 for validation when left unset; the primary Prediction control owns prediction confidence.",
    "copy_paste": "Probability of segmentation copy-paste augmentation. It is not used by axis-aligned Detect training.",
    "cos_lr": "Use a cosine learning-rate schedule instead of the default schedule during training.",
    "crop_fraction": "Classification evaluation/inference crop fraction. It does not affect Detect training or prediction.",
    "data": "Upstream dataset-definition input. Normal Workbench training manages it from the prepared dataset, and normal prediction does not use it.",
    "degrees": "Maximum absolute rotation, in degrees, used by image augmentation; a value of 0 disables rotation.",
    "deterministic": "Request deterministic algorithms for more reproducible training where the installed PyTorch/CUDA stack supports them. It can reduce throughput.",
    "dfl": "Gain applied to Distribution Focal Loss, the bounding-box distribution term used by supported Detect models during training.",
    "dnn": "Use OpenCV DNN for ONNX inference. It has no effect when predicting with the normal PyTorch .pt workflow.",
    "dropout": "Dropout regularization rate for classification training only. It does not affect Detect training.",
    "dynamic": "Export-only switch for dynamic input axes in ONNX, TensorFlow, or TensorRT exports. It is not used by training or normal prediction.",
    "embed": "Optional list of model layer indices from which to return embeddings. It is an upstream prediction API feature and is not displayed as a Workbench result.",
    "epochs": "Maximum number of training epochs. The primary Train control owns this value; it is not used by prediction.",
    "erasing": "Probability of random erasing for classification training only. It does not affect Detect training.",
    "fliplr": "Probability of a left-right image flip during augmentation.",
    "flipud": "Probability of an up-down image flip during augmentation.",
    "format": "Target format for the upstream export mode, such as torchscript. The Workbench does not run export jobs, so it has no effect here.",
    "fraction": "Fraction of the training split to use, from 0 to 1. A value of 1.0 uses all training images.",
    "freeze": "Optional number of initial model layers, or explicit layer indices, to freeze during training. Frozen layers are not updated.",
    "half": "Use FP16 inference/validation where the selected device and model support it. It may improve throughput on compatible CUDA hardware.",
    "hsv_h": "Maximum hue shift, expressed as a fraction, for HSV color augmentation.",
    "hsv_s": "Maximum saturation shift, expressed as a fraction, for HSV color augmentation.",
    "hsv_v": "Maximum value/brightness shift, expressed as a fraction, for HSV color augmentation.",
    "int8": "Export-only request for INT8 quantization in supported CoreML or TensorFlow exports. It is not used by training or normal prediction.",
    "iou": "Intersection-over-union threshold used by non-maximum suppression. Higher values retain more overlapping detections; the primary Prediction control owns its prediction value.",
    "keras": "Export-only request to use Keras. It is not used by the Workbench's Detect training or .pt prediction workflow.",
    "kobj": "Gain applied to keypoint-objectness loss for pose models. It does not affect Detect training.",
    "label_smoothing": "Amount of label smoothing used for classification targets during training; 0 disables it.",
    "line_width": "Optional bounding-box line width in pixels for saved or displayed predictions. When unset, Ultralytics scales it from the image size.",
    "lr0": "Initial learning rate. Ultralytics notes that typical starting values differ by optimizer (for example, SGD 1e-2 and Adam 1e-3).",
    "lrf": "Final learning-rate multiplier: the final rate is lr0 multiplied by lrf.",
    "mask_ratio": "Segmentation-mask downsample ratio for segmentation training only. It does not affect Detect training.",
    "max_det": "Maximum number of detections retained for each image after prediction or validation.",
    "mixup": "Probability of MixUp augmentation, which blends pairs of training images and labels.",
    "momentum": "SGD momentum or Adam-family beta1 value used by the optimizer.",
    "mosaic": "Probability of mosaic augmentation, which combines multiple images into one training sample.",
    "multi_scale": "Randomly vary the training image size around imgsz. It is a training-only performance/augmentation trade-off.",
    "nbs": "Nominal batch size used by Ultralytics to scale optimizer hyperparameters. It is not the actual batch selected in the primary Train control.",
    "nms": "Export-only option to add non-maximum suppression to a CoreML export. It is not used by training or normal prediction.",
    "opset": "Optional ONNX opset version for export. It is not used by training or normal prediction.",
    "optimize": "Export-only TorchScript optimization for mobile. It is not used by training or normal prediction.",
    "optimizer": "Optimizer choice: SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, or auto. Auto lets Ultralytics choose.",
    "overlap_mask": "Allow overlapping masks during segmentation training. It does not affect Detect training.",
    "patience": "Number of epochs with no observed improvement before early stopping. The primary Train control owns this value; 0 disables patience-based stopping.",
    "perspective": "Maximum perspective transform magnitude for augmentation. Ultralytics expects a small value, typically in the 0 to 0.001 range.",
    "plots": "Save plots and images produced during training or validation. This controls generated diagnostics, not live browser charts.",
    "pose": "Gain applied to pose loss for pose models. It does not affect Detect training.",
    "pretrained": "Use pretrained weights, or provide a weight path to load. The primary Model control owns the standard Workbench selection.",
    "profile": "Profile ONNX and TensorRT speeds for supported loggers during training. It is not a normal prediction-speed switch.",
    "rect": "Use rectangular batches for training, or rectangular validation in validation mode. It can reduce padding but changes batching behavior.",
    "resume": "Resume a compatible interrupted training run from its last checkpoint. It is training-only and requires the checkpoint to remain available.",
    "retina_masks": "Use high-resolution segmentation masks. It is a segmentation-only prediction control and does not affect Detect boxes.",
    "save": "Save training checkpoints or prediction results. The Workbench manages saving for normal runs, so changing this is not needed.",
    "save_conf": "Include confidence scores when saving text prediction labels. It only matters when save_txt is enabled.",
    "save_crop": "Save a separate crop for each detected object. It creates additional output files.",
    "save_frames": "Save every processed video frame as an image during prediction. It can create a large number of files.",
    "save_hybrid": "Save hybrid validation labels containing ground-truth labels plus additional predictions. It is a validation-oriented upstream option.",
    "save_json": "Save validation results in COCO-style JSON where applicable. It is mainly useful for compatible evaluation workflows.",
    "save_period": "Save a training checkpoint every N epochs. Values below 1 disable periodic checkpoint saving.",
    "save_txt": "Save prediction results as text label files alongside visual output.",
    "scale": "Maximum relative image scaling gain for augmentation; a value of 0 disables scale augmentation.",
    "seed": "Random seed used for reproducible training data order and augmentations where the underlying stack permits it.",
    "shear": "Maximum absolute shear angle, in degrees, for image augmentation; a value of 0 disables shearing.",
    "show": "Ask Ultralytics to display predicted images or videos in a local viewer when the environment allows. It may have no visible effect in a server process.",
    "show_boxes": "Draw detection boxes in rendered prediction output.",
    "show_conf": "Draw confidence values in rendered prediction output.",
    "show_labels": "Draw class labels in rendered prediction output.",
    "simplify": "Export-only request to simplify an ONNX model with onnxslim. It is not used by training or normal prediction.",
    "single_cls": "Treat all dataset labels as one class during training. This changes the learning target and is training-only.",
    "source": "Upstream prediction source input. The primary Prediction source control owns it; it is not used by training.",
    "split": "Dataset split used for validation: train, val, or test. It is relevant to validation, not normal prediction.",
    "stream_buffer": "For streaming sources, true buffers every frame and false keeps only the most recent frame. It has no practical effect for finite local files.",
    "time": "Optional training time limit in hours. When supplied, it overrides the epoch limit; it is not used by prediction.",
    "tracker": "Tracker configuration file for the upstream track mode, such as botsort.yaml. The Workbench does not run tracking jobs.",
    "translate": "Maximum absolute horizontal and vertical translation, expressed as a fraction of image size, for augmentation.",
    "val": "Run validation during training. It is a training-only control.",
    "val_period": "Run validation every N training epochs. It is a training-only control.",
    "verbose": "Request verbose Ultralytics console output. The Workbench manages ordinary run logging and prediction does not need this setting.",
    "vid_stride": "Process every Nth video frame during prediction. A value of 1 processes every frame.",
    "visualize": "Visualize model feature maps during prediction. It is a diagnostic option that may create additional output and slow inference.",
    "warmup_bias_lr": "Initial learning rate for bias parameters during the warmup phase.",
    "warmup_epochs": "Number of warmup epochs; fractional values are allowed.",
    "warmup_momentum": "Initial momentum used during warmup before it transitions to the configured momentum.",
    "weight_decay": "L2-style weight decay applied by the optimizer to regularize learned weights.",
    "workers": "Number of data-loader worker processes per DDP rank. The primary Train control owns the ordinary training value.",
    "workspace": "TensorRT export workspace size in GB. It is an export-only setting and is not used by training or normal prediction.",
}


def parameter_docs(mode: str) -> List[Dict[str, Any]]:
    """Return one documented row for every advanced field rendered in the UI."""
    rows: List[Dict[str, Any]] = []
    for group, fields in expert_groups(mode):
        for key, default in fields:
            rows.append({
                "group": group,
                "key": key,
                "default": repr(default),
                "type": "optional" if default is None else type(default).__name__,
                "description": PARAMETER_OVERRIDES[key],
            })
    return rows


def docs_page(slug: str) -> Dict[str, Any]:
    return {"slug": slug, **PAGE_META[slug]}


def docs_slugs() -> set[str]:
    return set(PAGE_META)


def _docs_url(slug: str) -> str:
    return "/docs" if slug == "getting-started" else f"/docs/{slug}"


def docs_search_index() -> List[Dict[str, str]]:
    """Build the small, client-side Docs index from authoritative page metadata."""
    navigation_titles = {
        page["slug"]: page["title"]
        for section in DOC_NAVIGATION
        for page in section["pages"]
    }
    entries: List[Dict[str, str]] = []
    for slug, meta in PAGE_META.items():
        page_title = navigation_titles[slug]
        entries.append({
            "kind": "Page",
            "title": page_title,
            "page_title": page_title,
            "url": _docs_url(slug),
            "terms": " ".join((page_title, meta["title"], meta["summary"])),
        })
        for anchor, title in meta["toc"]:
            entries.append({
                "kind": "Section",
                "title": title,
                "page_title": page_title,
                "url": f"{_docs_url(slug)}#{anchor}",
                "terms": " ".join((page_title, meta["title"], title, *DOC_SECTION_KEYWORDS[(slug, anchor)])),
            })
    return entries

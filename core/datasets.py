"""Strict, local-only dataset inspection and YOLO detection preparation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PREPARED_ROOT = ROOT / "datasets" / "_prepared"
SPLIT_NAMES = ("train", "val", "test")
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
DEFAULT_SPLIT_OPTIONS = {"create_val": True, "create_test": True, "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}, "seed": 42}
YOLO_COORDINATE_TOLERANCE = 0.01
FORMAT_CATALOG = [
    {"id": "coco", "name": "COCO / COCO-MMDetection", "family": "JSON", "status": "supported", "rule": "images, annotations and categories with [x, y, width, height] boxes."},
    {"id": "createml", "name": "CreateML", "family": "JSON", "status": "supported", "rule": "CreateML annotation coordinates are converted from centre/size to pixel corners."},
    {"id": "pascal-voc", "name": "Pascal VOC", "family": "XML", "status": "supported", "rule": "Per-image XML bndbox coordinates are preserved."},
    {"id": "tensorflow-csv", "name": "TensorFlow Object Detection CSV", "family": "CSV", "status": "supported", "rule": "The explicit filename, size, class and xmin/ymin/xmax/ymax columns are required."},
    {"id": "retinanet-csv", "name": "RetinaNet Keras CSV", "family": "CSV", "status": "validated", "rule": "Only a recognised, explicitly headed bbox CSV is accepted."},
    {"id": "yolo-txt", "name": "YOLO TXT family", "family": "TXT", "status": "supported", "rule": "Darknet, Scaled-YOLOv4, YOLOv5/6/7/8/9/10/11/12/26 normalized class xc yc w h labels."},
    {"id": "keras-yolo", "name": "YOLO v3 Keras / YOLO v4 PyTorch", "family": "TXT", "status": "supported", "rule": "annotations.txt pixel boxes plus a class list are converted exactly."},
    {"id": "paligemma", "name": "PaliGemma detection JSONL", "family": "JSONL", "status": "validated", "rule": "Local-image detect records with exact <loc> tokens only."},
    {"id": "florence-openai", "name": "Florence 2 / OpenAI Object Detection", "family": "JSONL", "status": "validated", "rule": "Only version-pinned local-image detection grammar is accepted."},
    {"id": "yolo-obb", "name": "YOLOv8 Oriented Bounding Boxes", "family": "TXT", "status": "incompatible", "rule": "Four-corner oriented boxes cannot be losslessly converted to axis-aligned Detect boxes."},
    {"id": "multi-label", "name": "Multi-Label Classification CSV", "family": "CSV", "status": "incompatible", "rule": "Image-level labels do not contain detection boxes."},
]


class DatasetError(ValueError):
    """A user-actionable, non-lossless dataset preparation failure."""


@dataclass
class Box:
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    source: str = ""


@dataclass
class ImageRecord:
    path: Path
    width: int
    height: int
    split: str
    boxes: list[Box] = field(default_factory=list)


@dataclass
class YoloLayout:
    """Resolved YOLO image/label directories and metadata for one dataset root."""

    root: Path
    mode: str
    yaml_path: Optional[Path]
    splits: dict[str, dict[str, Optional[Path]]]
    classes: dict[int, str]
    warnings: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    missing_labels: int = 0
    clipped_boxes: int = 0
    clipped_examples: list[str] = field(default_factory=list)


def _safe_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _dimensions(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise DatasetError(f"Cannot read image: {path}")
    height, width = image.shape[:2]
    if not width or not height:
        raise DatasetError(f"Image has no dimensions: {path}")
    return width, height


def _split(path: Path) -> str:
    for item in path.parts:
        name = item.lower()
        if name == "train":
            return "train"
        if name in {"val", "valid", "validation"}:
            return "val"
        if name == "test":
            return "test"
    return "train"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ProgressCallback = Callable[[int, str], None]


def _progress(callback: Optional[ProgressCallback], percent: int, message: str) -> None:
    if callback:
        callback(max(0, min(100, percent)), message)


def _fingerprint(root: Path, options: Optional[dict[str, Any]] = None, progress: Optional[ProgressCallback] = None, start: int = 68, end: int = 82) -> str:
    digest = hashlib.sha256(str(root).encode())
    if options:
        digest.update(json.dumps(options, sort_keys=True, default=str).encode())
    files = sorted(item for item in root.rglob("*") if item.is_file())
    for index, path in enumerate(files, 1):
        relative = path.relative_to(root).as_posix()
        metadata = path.stat()
        digest.update(relative.encode())
        digest.update(f"{metadata.st_size}:{metadata.st_mtime_ns}".encode())
        _progress(progress, start + int(index / max(1, len(files)) * (end - start)), f"Verifying source files ({index:,} of {len(files):,})")
    return digest.hexdigest()[:20]


def _relative_path(root: Path, path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _path_inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _yaml_metadata(root: Path) -> tuple[Optional[Path], dict[str, Any], list[str]]:
    candidates = sorted(
        [path for path in root.rglob("data.yaml") if path.is_file()] + [path for path in root.rglob("data.yml") if path.is_file()],
        key=lambda item: (len(item.relative_to(root).parts), item.as_posix()),
    )
    warnings: list[str] = []
    for path in candidates:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            warnings.append(f"Could not read {path.name}: {exc}")
            continue
        if isinstance(payload, dict):
            return path, payload, warnings
    return None, {}, warnings


def _classes_from_yaml(root: Path, yaml_path: Optional[Path] = None) -> dict[int, str]:
    paths = [yaml_path] if yaml_path else []
    if not paths:
        paths = sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml"))
    for path in paths:
        if not path or not path.is_file():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        names = payload.get("names") if isinstance(payload, dict) else None
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        if isinstance(names, dict):
            try:
                return {int(index): str(name) for index, name in names.items()}
            except (TypeError, ValueError):
                continue
    for name in ("classes.txt", "obj.names"):
        path = root / name
        if path.is_file():
            values = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
            return {index: value for index, value in enumerate(values)}
    return {}


def _normalise_split(value: str) -> Optional[str]:
    return SPLIT_ALIASES.get(value.strip().lower())


def _yaml_path(value: Any, yaml_path: Path, payload: dict[str, Any]) -> Optional[Path]:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if not isinstance(value, str) or not value.strip():
        return None
    base = yaml_path.parent
    yaml_base = payload.get("path")
    if isinstance(yaml_base, str) and yaml_base.strip():
        base = (base / yaml_base).resolve()
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _label_dir_for_images(images_dir: Path) -> Optional[Path]:
    if images_dir.name.lower() == "images":
        sibling = images_dir.parent / "labels"
        if sibling.is_dir():
            return sibling.resolve()
    for part_index, part in enumerate(images_dir.parts):
        if part.lower() == "images":
            sibling = Path(*images_dir.parts[:part_index], "labels", *images_dir.parts[part_index + 1:])
            if sibling.is_dir():
                return sibling.resolve()
    return None


def _manual_layout(root: Path, options: dict[str, Any], yaml_path: Optional[Path], classes: dict[int, str]) -> YoloLayout:
    raw_splits = options.get("splits") or {}
    if not isinstance(raw_splits, dict):
        raise DatasetError("Manual dataset mapping must contain split directories.")
    splits: dict[str, dict[str, Optional[Path]]] = {}
    for raw_name, values in raw_splits.items():
        split = _normalise_split(str(raw_name))
        if not split or not isinstance(values, dict):
            continue
        image_value = values.get("images")
        label_value = values.get("labels")
        if not image_value:
            continue
        image_dir = (root / str(image_value)).resolve() if not Path(str(image_value)).is_absolute() else Path(str(image_value)).resolve()
        label_dir = None
        if label_value:
            label_dir = (root / str(label_value)).resolve() if not Path(str(label_value)).is_absolute() else Path(str(label_value)).resolve()
        if not _path_inside(root, image_dir) or (label_dir and not _path_inside(root, label_dir)):
            raise DatasetError("Manual dataset directories must stay inside the selected dataset folder.")
        if not image_dir.is_dir():
            raise DatasetError(f"Image directory does not exist: {_relative_path(root, image_dir)}")
        if label_dir and not label_dir.is_dir():
            raise DatasetError(f"Label directory does not exist: {_relative_path(root, label_dir)}")
        splits[split] = {"images": image_dir, "labels": label_dir}
    if not splits:
        raise DatasetError("Choose at least one image directory in the dataset mapping.")
    return YoloLayout(root, "manual", yaml_path, splits, classes, candidates=[])


def _discover_layout_at(root: Path) -> Optional[YoloLayout]:
    yaml_path, payload, warnings = _yaml_metadata(root)
    classes = _classes_from_yaml(root, yaml_path)
    actual: dict[str, dict[str, Optional[Path]]] = {}
    candidates: list[dict[str, Any]] = []

    # Roboflow's split-first structure: train/images + train/labels.
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        if not child.is_dir():
            continue
        split = _normalise_split(child.name)
        images_dir = child / "images"
        labels_dir = child / "labels"
        if split and images_dir.is_dir():
            actual[split] = {"images": images_dir.resolve(), "labels": labels_dir.resolve() if labels_dir.is_dir() else None}

    # Ultralytics' images-first structure: images/train + labels/train.
    if not actual and (root / "images").is_dir():
        for child in sorted((root / "images").iterdir()):
            if not child.is_dir():
                continue
            split = _normalise_split(child.name)
            if split:
                labels_dir = root / "labels" / child.name
                actual[split] = {"images": child.resolve(), "labels": labels_dir.resolve() if labels_dir.is_dir() else None}
    if not actual and (root / "images").is_dir():
        labels_dir = root / "labels"
        actual["train"] = {"images": (root / "images").resolve(), "labels": labels_dir.resolve() if labels_dir.is_dir() else None}
    # Flat exports put the image and label files directly in the selected root.
    # Require a direct image so an uploaded archive wrapper containing a real
    # dataset directory is not mistaken for the dataset itself.
    if not actual and any(path.parent == root for path in _images(root)) and list(root.rglob("*.txt")):
        actual["train"] = {"images": root.resolve(), "labels": None}

    yaml_splits: dict[str, dict[str, Optional[Path]]] = {}
    if yaml_path:
        for key in ("train", "val", "valid", "test"):
            split = _normalise_split(key)
            if split in yaml_splits or key not in payload:
                continue
            image_dir = _yaml_path(payload.get(key), yaml_path, payload)
            if image_dir and image_dir.is_dir():
                label_dir = _label_dir_for_images(image_dir)
                yaml_splits[split] = {"images": image_dir, "labels": label_dir}

    splits = actual or yaml_splits
    if actual and yaml_splits:
        for split, mapping in yaml_splits.items():
            selected = actual.get(split)
            if selected and selected["images"] != mapping["images"]:
                warnings.append(f"data.yaml {split} path differs from the detected directory; the detected directory was used.")
    if not splits:
        return None
    if not any(mapping.get("labels") or list((mapping["images"] or root).rglob("*.txt")) for mapping in splits.values()):
        return None
    mode = "roboflow" if any(path.parent.name.lower() in {"train", "valid", "val", "validation", "test"} for path in (mapping["images"] for mapping in splits.values()) if path) else ("flat" if any(path == root for path in (mapping["images"] for mapping in splits.values()) if path) else "images-first")
    for split, mapping in splits.items():
        candidates.append({"split": split, "images": _relative_path(root, mapping.get("images")), "labels": _relative_path(root, mapping.get("labels"))})
    return YoloLayout(root, mode, yaml_path, splits, classes, warnings=warnings, candidates=candidates)


def _discover_yolo_layout(root: Path, options: Optional[dict[str, Any]] = None) -> Optional[YoloLayout]:
    options = options or {}
    yaml_path, _, warnings = _yaml_metadata(root)
    classes = _classes_from_yaml(root, yaml_path)
    if str(options.get("layout") or "").lower() == "manual":
        return _manual_layout(root, options, yaml_path, classes)
    candidates = [root]
    for path in sorted(root.rglob("data.yaml")) + sorted(root.rglob("data.yml")):
        if path.parent not in candidates and len(path.relative_to(root).parts) <= 3:
            candidates.append(path.parent)
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        if child.is_dir() and child not in candidates:
            candidates.append(child)
    discovered: list[YoloLayout] = []
    for candidate in candidates:
        layout = _discover_layout_at(candidate)
        if layout:
            discovered.append(layout)
    if not discovered:
        return None
    selected = max(discovered, key=lambda item: (len(item.splits), bool(item.yaml_path), -len(item.root.relative_to(root).parts)))
    if selected.root != root:
        selected.warnings.append(f"Using nested dataset root: {_relative_path(root, selected.root)}")
    selected.warnings = list(dict.fromkeys(warnings + selected.warnings))
    return selected


def _structure_summary(root: Path, max_depth: int = 3, max_entries: int = 40) -> dict[str, Any]:
    """Return a bounded tree summary without exposing every image filename."""
    nodes: list[dict[str, Any]] = []

    def visit(path: Path, depth: int) -> None:
        if len(nodes) >= max_entries:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        except OSError:
            return
        directories = [item for item in entries if item.is_dir()]
        files = [item for item in entries if item.is_file()]
        node = {"path": _relative_path(root, path) or ".", "directories": [item.name for item in directories[:12]], "samples": [item.name for item in files[:3]], "file_count": len(files), "omitted": max(0, len(files) - 3)}
        nodes.append(node)
        if depth >= max_depth:
            return
        for directory in directories[:12]:
            visit(directory, depth + 1)

    visit(root.resolve(), 0)
    return {"root": str(root.resolve()), "nodes": nodes, "max_depth": max_depth, "max_entries": max_entries, "truncated": len(nodes) >= max_entries}


def _layout_payload(layout: Optional[YoloLayout], root: Path) -> dict[str, Any]:
    if not layout:
        return {"mode": "unknown", "root": str(root), "yaml": "", "splits": {}, "candidates": [], "warnings": [], "clipped_boxes": 0}
    splits = {}
    for split, mapping in layout.splits.items():
        images_dir, labels_dir = mapping.get("images"), mapping.get("labels")
        splits[split] = {"images": _relative_path(root, images_dir), "labels": _relative_path(root, labels_dir), "images_count": len(_images(images_dir)) if images_dir else 0, "labels_count": len(list(labels_dir.rglob("*.txt"))) if labels_dir else 0}
    return {"mode": layout.mode, "root": str(layout.root), "yaml": _relative_path(root, layout.yaml_path), "splits": splits, "candidates": layout.candidates, "warnings": layout.warnings, "clipped_boxes": layout.clipped_boxes}


def _options_layout_payload(settings: dict[str, Any], root: Path) -> dict[str, Any]:
    """Keep user-entered manual paths visible when validation fails."""
    splits: dict[str, dict[str, Any]] = {}
    raw_splits = settings.get("splits") if isinstance(settings.get("splits"), dict) else {}
    for raw_name, values in raw_splits.items():
        split = _normalise_split(str(raw_name))
        if not split or not isinstance(values, dict):
            continue
        images = str(values.get("images") or "")
        labels = str(values.get("labels") or "")
        if images or labels:
            splits[split] = {"images": images, "labels": labels, "images_count": 0, "labels_count": 0}
    return {"mode": settings.get("layout") or "auto", "root": str(root), "yaml": "", "splits": splits, "candidates": [], "warnings": [], "clipped_boxes": 0}


def _find_image(root: Path, name: str) -> Path:
    candidate = Path(name)
    direct = candidate if candidate.is_absolute() else root / candidate
    if direct.is_file():
        return direct.resolve()
    matches = [path for path in _images(root) if path.name == candidate.name]
    if len(matches) != 1:
        raise DatasetError(f"Cannot uniquely locate image '{name}'.")
    return matches[0]


def _record(path: Path, boxes: list[Box], split: str | None = None, dimensions: tuple[int, int] | None = None) -> ImageRecord:
    width, height = dimensions or _dimensions(path)
    return ImageRecord(path=path, width=width, height=height, split=split or _split(path), boxes=boxes)


def _parse_yolo(root: Path, layout: Optional[YoloLayout] = None) -> list[ImageRecord]:
    layout = layout or _discover_yolo_layout(root)
    if not layout:
        raise DatasetError("Could not identify a YOLO image/label directory layout.")
    classes = layout.classes
    records: list[ImageRecord] = []
    label_count = 0
    obb_lines = 0
    seen_images: set[Path] = set()
    for split, mapping in layout.splits.items():
        images_dir = mapping.get("images")
        labels_dir = mapping.get("labels")
        if not images_dir:
            continue
        for image in _images(images_dir):
            image = image.resolve()
            if image in seen_images:
                continue
            seen_images.add(image)
            relative = image.relative_to(images_dir)
            candidates = [image.with_suffix(".txt")]
            if labels_dir:
                candidates.insert(0, labels_dir / relative.with_suffix(".txt"))
            label = next((path for path in candidates if path.is_file()), None)
            width, height = _dimensions(image)
            boxes: list[Box] = []
            if label:
                label_count += 1
                for number, raw in enumerate(label.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    values = raw.split()
                    if not values:
                        continue
                    if len(values) == 9:
                        obb_lines += 1
                        continue
                    if len(values) != 5:
                        raise DatasetError(f"{label}:{number} must contain class xc yc width height.")
                    try:
                        class_id, xc, yc, box_w, box_h = int(values[0]), *map(float, values[1:])
                    except ValueError as exc:
                        raise DatasetError(f"{label}:{number} has invalid numeric values.") from exc
                    coordinates = (xc, yc, box_w, box_h)
                    if (
                        class_id < 0
                        or not all(math.isfinite(value) for value in coordinates)
                        or max(coordinates) > 1 + YOLO_COORDINATE_TOLERANCE
                        or min(coordinates) < -YOLO_COORDINATE_TOLERANCE
                        or box_w <= 0
                        or box_h <= 0
                    ):
                        raise DatasetError(f"{label}:{number} has invalid normalized coordinates.")
                    source = f"{label}:{number}"
                    xmin, ymin = (xc - box_w / 2) * width, (yc - box_h / 2) * height
                    xmax, ymax = (xc + box_w / 2) * width, (yc + box_h / 2) * height
                    clipped = (max(0.0, min(float(width), xmin)), max(0.0, min(float(height), ymin)), max(0.0, min(float(width), xmax)), max(0.0, min(float(height), ymax)))
                    if clipped != (xmin, ymin, xmax, ymax):
                        layout.clipped_boxes += 1
                        if len(layout.clipped_examples) < 10:
                            layout.clipped_examples.append(source)
                    xmin, ymin, xmax, ymax = clipped
                    if not (xmin < xmax and ymin < ymax):
                        raise DatasetError(f"{source} becomes a zero-area box after clipping to the image boundary.")
                    boxes.append(Box(classes.get(class_id, f"class_{class_id}"), xmin, ymin, xmax, ymax, source))
            else:
                layout.missing_labels += 1
            records.append(ImageRecord(image, width, height, split, boxes))
    if obb_lines:
        raise DatasetError("Recognized YOLO oriented bounding boxes. OBB cannot be losslessly prepared for YOLOv10 Detect.")
    if not records:
        raise DatasetError("No images found in the selected YOLO image directories.")
    if not label_count:
        raise DatasetError("No YOLO label files were found for the selected images. Choose matching image and label directories.")
    return records


def _parse_coco(root: Path, annotation: Path) -> list[ImageRecord]:
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    if not {"images", "annotations", "categories"}.issubset(payload):
        raise DatasetError("COCO JSON needs images, annotations and categories arrays.")
    categories = {item["id"]: str(item["name"]) for item in payload["categories"]}
    grouped: dict[Any, list[Box]] = {}
    for item in payload["annotations"]:
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise DatasetError(f"{annotation}: annotation {item.get('id', '?')} has no axis-aligned bbox.")
        x, y, width, height = map(float, bbox)
        grouped.setdefault(item.get("image_id"), []).append(Box(categories.get(item.get("category_id"), f"class_{item.get('category_id')}"), x, y, x + width, y + height, f"{annotation}:annotation:{item.get('id', '?')}"))
    records = []
    for image in payload["images"]:
        path = _find_image(root, str(image.get("file_name", "")))
        width, height = int(image.get("width") or 0), int(image.get("height") or 0)
        actual = _dimensions(path)
        if not width or not height:
            width, height = actual
        if actual != (width, height):
            raise DatasetError(f"{annotation}: dimensions for {path.name} do not match image bytes.")
        records.append(_record(path, grouped.get(image.get("id"), []), dimensions=(width, height)))
    return records


def _parse_voc(root: Path, files: list[Path]) -> list[ImageRecord]:
    records = []
    for annotation in files:
        node = ET.parse(annotation).getroot()
        filename = (node.findtext("filename") or "").strip()
        path = _find_image(root, filename) if filename else next((candidate for candidate in _images(root) if candidate.stem == annotation.stem), None)
        if not path:
            raise DatasetError(f"{annotation}: no matching image.")
        width, height = _dimensions(path)
        boxes = []
        for object_node in node.findall("object"):
            name = (object_node.findtext("name") or "").strip()
            bbox = object_node.find("bndbox")
            if not name or bbox is None:
                raise DatasetError(f"{annotation}: object is missing name or bndbox.")
            try:
                xmin, ymin, xmax, ymax = (float(bbox.findtext(key, "")) for key in ("xmin", "ymin", "xmax", "ymax"))
            except ValueError as exc:
                raise DatasetError(f"{annotation}: invalid bndbox values.") from exc
            boxes.append(Box(name, xmin, ymin, xmax, ymax, str(annotation)))
        records.append(ImageRecord(path, width, height, _split(path), boxes))
    return records


def _parse_createml(root: Path, annotation: Path) -> list[ImageRecord]:
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise DatasetError("CreateML annotation file must contain a JSON array.")
    records = []
    for item in payload:
        path = _find_image(root, str(item.get("image", "")))
        width, height = _dimensions(path)
        boxes = []
        for value in item.get("annotations", []):
            coordinates = value.get("coordinates") or {}
            try:
                x, y, box_w, box_h = (float(coordinates[key]) for key in ("x", "y", "width", "height"))
            except (KeyError, TypeError, ValueError) as exc:
                raise DatasetError(f"{annotation}: invalid CreateML coordinates for {path.name}.") from exc
            boxes.append(Box(str(value.get("label") or ""), x - box_w / 2, y - box_h / 2, x + box_w / 2, y + box_h / 2, str(annotation)))
        records.append(ImageRecord(path, width, height, _split(path), boxes))
    return records


def _parse_csv(root: Path, annotation: Path) -> list[ImageRecord]:
    with annotation.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        headers = {value.lower().strip(): value for value in reader.fieldnames or []}
        canonical = {"filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"}
        if not canonical.issubset(headers):
            if {"image", "labels"}.issubset(headers) or "labels" in headers:
                raise DatasetError("Recognized multi-label classification CSV. It has no detection boxes for YOLOv10 Detect.")
            raise DatasetError("CSV must declare filename, width, height, class, xmin, ymin, xmax and ymax headers.")
        grouped: dict[str, list[Box]] = {}
        dimensions: dict[str, tuple[int, int]] = {}
        for number, row in enumerate(reader, 2):
            try:
                name = row[headers["filename"]]
                width, height = int(row[headers["width"]]), int(row[headers["height"]])
                box = Box(row[headers["class"]], *(float(row[headers[key]]) for key in ("xmin", "ymin", "xmax", "ymax")), source=f"{annotation}:{number}")
            except (KeyError, TypeError, ValueError) as exc:
                raise DatasetError(f"{annotation}:{number} has invalid detection CSV values.") from exc
            grouped.setdefault(name, []).append(box)
            dimensions[name] = (width, height)
    return [_record(_find_image(root, name), boxes, dimensions=dimensions[name]) for name, boxes in grouped.items()]


def _parse_keras(root: Path, annotation: Path) -> list[ImageRecord]:
    records = []
    for number, raw in enumerate(annotation.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        fields = raw.strip().split()
        if len(fields) < 2:
            continue
        path = _find_image(root, fields[0])
        boxes = []
        for item in fields[1:]:
            try:
                xmin, ymin, xmax, ymax, class_id = item.split(",")
                boxes.append(Box(f"class_{int(class_id)}", float(xmin), float(ymin), float(xmax), float(ymax), f"{annotation}:{number}"))
            except ValueError as exc:
                raise DatasetError(f"{annotation}:{number} has an invalid Keras YOLO box.") from exc
        records.append(_record(path, boxes))
    if not records:
        raise DatasetError("annotations.txt contains no YOLO Keras / PyTorch records.")
    return records


def _parse_paligemma(root: Path, annotation: Path) -> list[ImageRecord]:
    import re
    token = re.compile(r"<loc(\d{1,4})><loc(\d{1,4})><loc(\d{1,4})><loc(\d{1,4})>\s*([^<;]+)")
    records = []
    for number, raw in enumerate(annotation.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{annotation}:{number} is not valid JSONL.") from exc
        if not str(item.get("prefix", "")).lower().startswith("detect"):
            raise DatasetError(f"{annotation}:{number} is not a PaliGemma detection record.")
        path = _find_image(root, str(item.get("image", "")))
        width, height = _dimensions(path)
        matches = token.findall(str(item.get("suffix", "")))
        if not matches:
            raise DatasetError(f"{annotation}:{number} has no exact PaliGemma <loc> detection tokens.")
        boxes = [Box(label.strip(), int(x1) / 1024 * width, int(y1) / 1024 * height, int(x2) / 1024 * width, int(y2) / 1024 * height, f"{annotation}:{number}") for y1, x1, y2, x2, label in matches]
        records.append(ImageRecord(path, width, height, _split(path), boxes))
    return records


def _detect(root: Path, yolo_layout: Optional[YoloLayout] = None) -> tuple[str, Callable[[], list[ImageRecord]], str]:
    json_files = list(root.rglob("*.json"))
    xml_files = list(root.rglob("*.xml"))
    csv_files = list(root.rglob("*.csv"))
    jsonl_files = list(root.rglob("*.jsonl"))
    keras = next((path for path in root.rglob("annotations.txt") if path.is_file()), None)
    for path in json_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and {"images", "annotations", "categories"}.issubset(payload):
            return "coco", lambda path=path: _parse_coco(root, path), f"{path.name}: images, annotations and categories"
        if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "image" in payload[0] and "annotations" in payload[0]:
            return "createml", lambda path=path: _parse_createml(root, path), f"{path.name}: CreateML image/annotations records"
    if xml_files and any("annotation" in path.read_text(encoding="utf-8", errors="ignore")[:500] and "bndbox" in path.read_text(encoding="utf-8", errors="ignore")[:5000] for path in xml_files):
        return "pascal-voc", lambda: _parse_voc(root, xml_files), f"{len(xml_files)} Pascal VOC XML annotations"
    if keras:
        return "keras-yolo", lambda: _parse_keras(root, keras), "annotations.txt"
    if csv_files:
        return "tensorflow-csv", lambda path=csv_files[0]: _parse_csv(root, path), f"{csv_files[0].name}: CSV schema"
    if jsonl_files:
        first = jsonl_files[0]
        sample = first.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
        if sample and "<loc" in sample[0]:
            return "paligemma", lambda: _parse_paligemma(root, first), f"{first.name}: local <loc> detection tokens"
        raise DatasetError("Recognized JSONL, but its Florence/OpenAI grammar is not an exact local detection grammar supported by this version.")
    if yolo_layout:
        return "yolo-txt", lambda: _parse_yolo(root, yolo_layout), f"{yolo_layout.mode} YOLO image/label directories"
    labels = list(root.rglob("*.txt"))
    if labels and _images(root):
        return "yolo-txt", lambda: _parse_yolo(root), "image files with YOLO TXT labels"
    raise DatasetError("Could not identify a supported detection dataset format in this folder.")


def _validate(records: list[ImageRecord]) -> None:
    if not records:
        raise DatasetError("No annotated images were found.")
    hashes: dict[str, str] = {}
    check_cross_split_duplicates = len({record.split for record in records}) > 1
    for record in records:
        if not record.boxes:
            continue
        if check_cross_split_duplicates:
            digest = _sha256(record.path)
            prior = hashes.get(digest)
            if prior and prior != record.split:
                raise DatasetError(f"Image {record.path.name} is duplicated across {prior} and {record.split} splits.")
            hashes[digest] = record.split
        for box in record.boxes:
            if not box.class_name.strip() or not (0 <= box.xmin < box.xmax <= record.width and 0 <= box.ymin < box.ymax <= record.height):
                raise DatasetError(f"Invalid or out-of-bounds box in {box.source or record.path}.")


def _normalise_options(options: Optional[dict[str, Any]]) -> dict[str, Any]:
    raw = options if isinstance(options, dict) else {}
    split = raw.get("split") if isinstance(raw.get("split"), dict) else {}
    ratios = split.get("ratios") if isinstance(split.get("ratios"), dict) else {}
    explicit_split = bool(raw.get("_explicit_split", isinstance(raw.get("split"), dict)))
    result = {
        "layout": str(raw.get("layout") or "auto").lower(),
        "_explicit_split": explicit_split,
        "splits": raw.get("splits") if isinstance(raw.get("splits"), dict) else {},
        "classes": [str(value).strip() for value in raw.get("classes", []) if str(value).strip()] if isinstance(raw.get("classes"), list) else [],
        "split": {
            "create_val": bool(split.get("create_val", DEFAULT_SPLIT_OPTIONS["create_val"])),
            "create_test": bool(split.get("create_test", DEFAULT_SPLIT_OPTIONS["create_test"])),
            "ratios": {
                "train": float(ratios.get("train", DEFAULT_SPLIT_OPTIONS["ratios"]["train"])),
                "val": float(ratios.get("val", DEFAULT_SPLIT_OPTIONS["ratios"]["val"])),
                "test": float(ratios.get("test", DEFAULT_SPLIT_OPTIONS["ratios"]["test"])),
            },
            "seed": int(split.get("seed", DEFAULT_SPLIT_OPTIONS["seed"])),
        },
    }
    for key, value in result["split"]["ratios"].items():
        if value < 0:
            raise DatasetError(f"Split ratio '{key}' cannot be negative.")
    if not any(result["split"]["ratios"].values()):
        raise DatasetError("At least one split ratio must be greater than zero.")
    return result


def _allocate_counts(total: int, ratios: dict[str, float], enabled: list[str]) -> dict[str, int]:
    if not enabled:
        return {}
    if total < len(enabled):
        raise DatasetError(f"At least {len(enabled)} images are required to create {', '.join(enabled)} splits.")
    weight = sum(max(0.0, ratios.get(name, 0.0)) for name in enabled)
    if weight <= 0:
        raise DatasetError("Enabled split ratios must contain a positive value.")
    raw = {name: total * max(0.0, ratios.get(name, 0.0)) / weight for name in enabled}
    counts = {name: max(1, int(raw[name])) for name in enabled}
    while sum(counts.values()) > total:
        candidates = [name for name in enabled if counts[name] > 1]
        if not candidates:
            raise DatasetError("Split ratios leave too few images for the requested splits.")
        candidate = max(candidates, key=lambda name: (counts[name] - raw[name], counts[name]))
        counts[candidate] -= 1
    while sum(counts.values()) < total:
        candidate = max(enabled, key=lambda name: (raw[name] - counts[name], -enabled.index(name)))
        counts[candidate] += 1
    return counts


def _stratified_order(records: list[ImageRecord], seed: int) -> list[ImageRecord]:
    groups: dict[tuple[str, ...], list[ImageRecord]] = {}
    for record in records:
        signature = tuple(sorted({box.class_name for box in record.boxes})) or ("__empty__",)
        groups.setdefault(signature, []).append(record)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    ordered: list[ImageRecord] = []
    while groups:
        for signature in sorted(list(groups), key=lambda item: (len(groups[item]), item)):
            values = groups.get(signature)
            if not values:
                groups.pop(signature, None)
                continue
            ordered.append(values.pop())
            if not values:
                groups.pop(signature, None)
    return ordered


def _ensure_splits(records: list[ImageRecord], options: Optional[dict[str, Any]] = None, default_test: bool = True) -> dict[str, Any]:
    """Preserve declared splits and deterministically create missing splits in memory."""
    settings = _normalise_options(options)
    split = settings["split"]
    existing = {record.split for record in records}
    if "val" in existing:
        split["create_val"] = False
    if "test" in existing:
        split["create_test"] = False
    # When a source already has validation data, test creation is opt-in. For a train-only
    # export both missing splits are enabled by default.
    if not settings.get("_explicit_split"):
        split["create_test"] = default_test and "test" not in existing and "val" not in existing
    missing = []
    if "val" not in existing and split["create_val"]:
        missing.append("val")
    if "test" not in existing and split["create_test"]:
        missing.append("test")
    pool = [record for record in records if record.split == "train"]
    if missing:
        counts = _allocate_counts(len(pool), split["ratios"], ["train", *missing])
        ordered = _stratified_order(pool, split["seed"])
        cursor = counts["train"]
        for name in missing:
            for record in ordered[cursor:cursor + counts[name]]:
                record.split = name
            cursor += counts[name]
    counts = {name: sum(1 for record in records if record.split == name) for name in SPLIT_NAMES}
    return {"ratios": split["ratios"], "create_val": split["create_val"], "create_test": split["create_test"], "seed": split["seed"], "counts": counts, "generated": missing}


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_prepared(
    root: Path,
    fingerprint: str,
    fmt: str,
    evidence: str,
    records: list[ImageRecord],
    progress: Optional[ProgressCallback] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    target = PREPARED_ROOT / fingerprint
    manifest = target / "conversion-report.json"
    if manifest.is_file():
        return target
    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{fingerprint}-", dir=PREPARED_ROOT))
    try:
        names = sorted({box.class_name for record in records for box in record.boxes})
        class_ids = {name: index for index, name in enumerate(names)}
        rows: list[dict[str, Any]] = []
        split_counts = {"train": 0, "val": 0, "test": 0}
        for index, record in enumerate(records):
            split = record.split
            split_counts.setdefault(split, 0)
            output_name = f"{index:06d}_{record.path.name}"
            destination = temp / "images" / split / output_name
            _link_or_copy(record.path, destination)
            label = temp / "labels" / split / f"{Path(output_name).stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for box in record.boxes:
                xc = ((box.xmin + box.xmax) / 2) / record.width
                yc = ((box.ymin + box.ymax) / 2) / record.height
                width = (box.xmax - box.xmin) / record.width
                height = (box.ymax - box.ymin) / record.height
                lines.append(f"{class_ids[box.class_name]} {xc:.10g} {yc:.10g} {width:.10g} {height:.10g}")
            label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            split_counts[split] += 1
            source_stat = record.path.stat()
            rows.append({"source": str(record.path), "size": source_stat.st_size, "modified_ns": source_stat.st_mtime_ns, "split": split, "output": str(destination.relative_to(temp)), "boxes": len(record.boxes)})
            _progress(progress, 83 + int((index + 1) / max(1, len(records)) * 16), f"Preparing dataset files ({index + 1:,} of {len(records):,})")
        with (temp / "data.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump({"path": str(target), "train": "images/train", "val": "images/val", "test": "images/test", "names": names}, stream, sort_keys=False, allow_unicode=True)
        report = {"fingerprint": fingerprint, "format": fmt, "evidence": evidence, "status": "ready", "classes": names, "splits": split_counts, "images": len(records), "objects": sum(len(item.boxes) for item in records), "source": str(root), "source_unchanged": True, "records": rows, **(metadata or {})}
        (temp / "conversion-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (temp / "source-manifest.json").write_text(json.dumps({"source": str(root), "fingerprint": fingerprint, "files": rows}, indent=2), encoding="utf-8")
        try:
            temp.replace(target)
        except FileExistsError:
            shutil.rmtree(temp, ignore_errors=True)
        return target
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def inspect_dataset(path_value: str, prepare: bool = True, progress: Optional[ProgressCallback] = None, options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Inspect and, when requested, create a strict cache-backed YOLO Detect dataset."""
    if not path_value.strip():
        return {"status": "neutral", "message": "Add a dataset folder to inspect it."}
    settings: dict[str, Any] = {}
    root: Optional[Path] = None
    context: dict[str, Any] = {}
    try:
        settings = _normalise_options(options)
        _progress(progress, 4, "Checking dataset folder")
        root = _safe_path(path_value)
        if not root.is_dir():
            if "\\" in path_value and ":" in path_value:
                raise DatasetError("This looks like a client Windows path. Enter a folder on the WebUI server or upload the dataset ZIP through this page.")
            raise DatasetError("Dataset folder was not found or is not a directory.")
        context["structure"] = _structure_summary(root)
        context["layout"] = _options_layout_payload(settings, root)
        context["class_names"] = list(settings.get("classes") or [])
        context["options_json"] = json.dumps(settings, sort_keys=True)
        _progress(progress, 16, "Identifying annotation format")
        layout = _discover_yolo_layout(root, settings)
        if layout and settings.get("classes"):
            layout.classes = {index: name for index, name in enumerate(settings["classes"])}
        context["layout"] = _layout_payload(layout, root)
        context["class_names"] = [name for _, name in sorted(layout.classes.items())] if layout else []
        if layout and layout.missing_labels:
            context["layout"]["missing_labels"] = layout.missing_labels
        fmt, parser, evidence = _detect(root, layout)
        _progress(progress, 18, "Checking prepared dataset cache")
        fingerprint = _fingerprint(root, settings, progress, 18, 30)
        cached_report = PREPARED_ROOT / fingerprint / "conversion-report.json"
        if prepare and cached_report.is_file():
            report = json.loads(cached_report.read_text(encoding="utf-8"))
            _progress(progress, 100, "Reused prepared dataset")
            return {"status": "ready", "message": "Dataset is ready for YOLOv10 Detect.", "format": fmt, "evidence": evidence, "prepared_path": str(cached_report.parent / "data.yaml"), **context, **report}
        _progress(progress, 36, "Reading images and annotations")
        records = parser()
        if layout:
            context["layout"]["missing_labels"] = layout.missing_labels
            context["layout"]["clipped_boxes"] = layout.clipped_boxes
        annotation_repairs = {
            "clipped_boxes": layout.clipped_boxes if layout else 0,
            "examples": layout.clipped_examples if layout else [],
            "policy": "Ultralytics normalized-coordinate tolerance and boundary clipping" if layout and layout.clipped_boxes else "",
        }
        context["annotation_repairs"] = annotation_repairs
        if not context.get("class_names"):
            context["class_names"] = sorted({box.class_name for record in records for box in record.boxes})
        _progress(progress, 64, f"Validating {len(records):,} image records")
        split_report = _ensure_splits(records, settings, default_test=fmt == "yolo-txt")
        _validate(records)
        _progress(progress, 82, "Creating prepared dataset")
        metadata = {"layout": context.get("layout", {}), "split_policy": split_report, "annotation_repairs": annotation_repairs}
        target = _write_prepared(root, fingerprint, fmt, evidence, records, progress, metadata) if prepare else None
        report = json.loads((target / "conversion-report.json").read_text(encoding="utf-8")) if target else {}
        _progress(progress, 100, "Dataset ready")
        return {"status": "ready", "message": "Dataset is ready for YOLOv10 Detect.", "format": fmt, "evidence": evidence, "prepared_path": str(target / "data.yaml") if target else "", **context, "split_policy": split_report, **report}
    except (DatasetError, OSError, json.JSONDecodeError, ET.ParseError) as exc:
        _progress(progress, 100, "Dataset preparation blocked")
        if root and "structure" not in context and root.is_dir():
            context["structure"] = _structure_summary(root)
        return {"status": "blocked", "message": str(exc), "prepared_path": "", **context, "manual_required": True}


def prepared_dataset(path_value: str) -> str:
    result = inspect_dataset(path_value, prepare=True)
    if result.get("status") != "ready":
        raise DatasetError(str(result.get("message") or "Dataset preparation failed."))
    return str(result["prepared_path"])

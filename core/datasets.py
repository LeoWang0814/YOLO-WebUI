"""Strict, local-only dataset inspection and YOLO detection preparation."""

from __future__ import annotations

import csv
import hashlib
import json
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


def _fingerprint(root: Path, progress: Optional[ProgressCallback] = None, start: int = 68, end: int = 82) -> str:
    digest = hashlib.sha256(str(root).encode())
    files = sorted(item for item in root.rglob("*") if item.is_file())
    for index, path in enumerate(files, 1):
        relative = path.relative_to(root).as_posix()
        metadata = path.stat()
        digest.update(relative.encode())
        digest.update(f"{metadata.st_size}:{metadata.st_mtime_ns}".encode())
        _progress(progress, start + int(index / max(1, len(files)) * (end - start)), f"Verifying source files ({index:,} of {len(files):,})")
    return digest.hexdigest()[:20]


def _classes_from_yaml(root: Path) -> dict[int, str]:
    for path in root.rglob("*.yaml"):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        names = payload.get("names")
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(index): str(name) for index, name in names.items()}
    for name in ("classes.txt", "obj.names"):
        path = root / name
        if path.is_file():
            values = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
            return {index: value for index, value in enumerate(values)}
    return {}


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


def _parse_yolo(root: Path) -> list[ImageRecord]:
    images = _images(root)
    if not images:
        raise DatasetError("No images found.")
    classes = _classes_from_yaml(root)
    records: list[ImageRecord] = []
    label_count = 0
    obb_lines = 0
    for image in images:
        relative = image.relative_to(root)
        label_relative = relative.with_suffix(".txt")
        if relative.parts and relative.parts[0].lower() == "images":
            label_relative = Path(*relative.parts[1:]).with_suffix(".txt")
        candidates = [image.with_suffix(".txt"), root / "labels" / label_relative]
        label = next((path for path in candidates if path.is_file()), None)
        if not label:
            continue
        label_count += 1
        width, height = _dimensions(image)
        boxes: list[Box] = []
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
            if class_id < 0 or min(xc, yc, box_w, box_h) < 0 or max(xc, yc, box_w, box_h) > 1:
                raise DatasetError(f"{label}:{number} has invalid normalized coordinates.")
            boxes.append(Box(classes.get(class_id, f"class_{class_id}"), (xc - box_w / 2) * width, (yc - box_h / 2) * height, (xc + box_w / 2) * width, (yc + box_h / 2) * height, f"{label}:{number}"))
        records.append(ImageRecord(image, width, height, _split(image), boxes))
    if obb_lines:
        raise DatasetError("Recognized YOLO oriented bounding boxes. OBB cannot be losslessly prepared for YOLOv10 Detect.")
    if not label_count:
        raise DatasetError("No YOLO label files were found next to images or under labels/.")
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


def _detect(root: Path) -> tuple[str, Callable[[], list[ImageRecord]], str]:
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


def _ensure_splits(records: list[ImageRecord]) -> None:
    """Create stable 80/10/10 splits only when the source declared none."""
    if any(record.split in {"val", "test"} for record in records):
        return
    if len(records) < 2:
        raise DatasetError("At least two images are required when no train/validation split is declared.")
    ordered = sorted(records, key=lambda record: _sha256(record.path))
    val_count = 1
    test_count = 1 if len(records) >= 6 else 0
    train_count = len(records) - val_count - test_count
    for index, record in enumerate(ordered):
        record.split = "train" if index < train_count else "val" if index < train_count + val_count else "test"


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_prepared(root: Path, fingerprint: str, fmt: str, evidence: str, records: list[ImageRecord], progress: Optional[ProgressCallback] = None) -> Path:
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
        report = {"fingerprint": fingerprint, "format": fmt, "evidence": evidence, "status": "ready", "classes": names, "splits": split_counts, "images": len(records), "objects": sum(len(item.boxes) for item in records), "source": str(root), "source_unchanged": True, "records": rows}
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


def inspect_dataset(path_value: str, prepare: bool = True, progress: Optional[ProgressCallback] = None) -> dict[str, Any]:
    """Inspect and, when requested, create a strict cache-backed YOLO Detect dataset."""
    if not path_value.strip():
        return {"status": "neutral", "message": "Add a dataset folder to inspect it."}
    try:
        _progress(progress, 4, "Checking dataset folder")
        root = _safe_path(path_value)
        if not root.is_dir():
            raise DatasetError("Dataset folder was not found or is not a directory.")
        _progress(progress, 16, "Identifying annotation format")
        fmt, parser, evidence = _detect(root)
        _progress(progress, 18, "Checking prepared dataset cache")
        fingerprint = _fingerprint(root, progress, 18, 30)
        cached_report = PREPARED_ROOT / fingerprint / "conversion-report.json"
        if prepare and cached_report.is_file():
            report = json.loads(cached_report.read_text(encoding="utf-8"))
            _progress(progress, 100, "Reused prepared dataset")
            return {"status": "ready", "message": "Dataset is ready for YOLOv10 Detect.", "format": fmt, "evidence": evidence, "prepared_path": str(cached_report.parent / "data.yaml"), **report}
        _progress(progress, 36, "Reading images and annotations")
        records = parser()
        _progress(progress, 64, f"Validating {len(records):,} image records")
        _ensure_splits(records)
        _validate(records)
        _progress(progress, 82, "Creating prepared dataset")
        target = _write_prepared(root, fingerprint, fmt, evidence, records, progress) if prepare else None
        report = json.loads((target / "conversion-report.json").read_text(encoding="utf-8")) if target else {}
        _progress(progress, 100, "Dataset ready")
        return {"status": "ready", "message": "Dataset is ready for YOLOv10 Detect.", "format": fmt, "evidence": evidence, "prepared_path": str(target / "data.yaml") if target else "", **report}
    except (DatasetError, OSError, json.JSONDecodeError, ET.ParseError) as exc:
        _progress(progress, 100, "Dataset preparation blocked")
        return {"status": "blocked", "message": str(exc), "prepared_path": ""}


def prepared_dataset(path_value: str) -> str:
    result = inspect_dataset(path_value, prepare=True)
    if result.get("status") != "ready":
        raise DatasetError(str(result.get("message") or "Dataset preparation failed."))
    return str(result["prepared_path"])

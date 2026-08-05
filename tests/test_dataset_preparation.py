import json

import cv2
import numpy as np

from core import datasets


def _image(path, width=100, height=80, value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((height, width, 3), value, dtype=np.uint8))


def test_yolo_folder_is_prepared_without_changing_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _image(source / "images" / "train" / "cat.jpg")
    _image(source / "images" / "val" / "dog.jpg", value=20)
    labels = source / "labels" / "train"
    labels.mkdir(parents=True)
    (labels / "cat.txt").write_text("0 0.5 0.5 0.4 0.5\n", encoding="utf-8")
    val_labels = source / "labels" / "val"
    val_labels.mkdir(parents=True)
    (val_labels / "dog.txt").write_text("0 0.5 0.5 0.4 0.5\n", encoding="utf-8")
    (source / "data.yaml").write_text("names: [cat]\n", encoding="utf-8")
    monkeypatch.setattr(datasets, "PREPARED_ROOT", tmp_path / "prepared")

    updates = []
    result = datasets.inspect_dataset(str(source), progress=lambda percent, message: updates.append((percent, message)))

    assert result["status"] == "ready", result
    assert result["format"] == "yolo-txt"
    prepared = tmp_path / "prepared" / result["fingerprint"]
    assert (prepared / "data.yaml").is_file()
    assert (prepared / "labels" / "train" / "000000_cat.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.4 0.5\n"
    assert (labels / "cat.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.4 0.5\n"
    assert updates[-1] == (100, "Dataset ready")
    assert any("Preparing dataset files" in message for _, message in updates)
    reused = []
    second = datasets.inspect_dataset(str(source), progress=lambda percent, message: reused.append((percent, message)))
    assert second["status"] == "ready"
    assert reused[-1] == (100, "Reused prepared dataset")


def test_coco_boxes_are_converted_from_xywh(tmp_path, monkeypatch):
    source = tmp_path / "coco"
    _image(source / "cat.jpg")
    _image(source / "dog.jpg", value=20)
    (source / "instances.json").write_text(json.dumps({"images": [{"id": 1, "file_name": "cat.jpg", "width": 100, "height": 80}, {"id": 2, "file_name": "dog.jpg", "width": 100, "height": 80}], "categories": [{"id": 5, "name": "cat"}], "annotations": [{"id": 7, "image_id": 1, "category_id": 5, "bbox": [10, 20, 40, 20]}, {"id": 8, "image_id": 2, "category_id": 5, "bbox": [10, 20, 40, 20]}]}), encoding="utf-8")
    monkeypatch.setattr(datasets, "PREPARED_ROOT", tmp_path / "prepared")

    result = datasets.inspect_dataset(str(source))

    assert result["status"] == "ready", result
    assert result["format"] == "coco"
    label = next((tmp_path / "prepared").rglob("*.txt"))
    assert label.read_text(encoding="utf-8") == "0 0.3 0.375 0.4 0.25\n"


def test_obb_and_multilabel_sources_are_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "PREPARED_ROOT", tmp_path / "prepared")
    obb = tmp_path / "obb"
    _image(obb / "cat.jpg")
    (obb / "cat.txt").write_text("0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n", encoding="utf-8")
    multi = tmp_path / "multi"
    _image(multi / "cat.jpg")
    (multi / "labels.csv").write_text("image,labels\ncat.jpg,cat;pet\n", encoding="utf-8")

    assert "oriented bounding" in datasets.inspect_dataset(str(obb))["message"].lower()
    assert "multi-label" in datasets.inspect_dataset(str(multi))["message"].lower()

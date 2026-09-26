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


def test_roboflow_split_first_layout_is_detected(tmp_path, monkeypatch):
    source = tmp_path / "roboflow"
    for split, count, value in (("train", 4, 0), ("valid", 2, 40), ("test", 2, 80)):
        for index in range(count):
            image = source / split / "images" / f"{split}-{index}.jpg"
            _image(image, value=value + index)
            label = source / split / "labels" / f"{split}-{index}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (source / "data.yaml").write_text("names: [drone]\ntrain: train/images\nval: valid/images\ntest: test/images\n", encoding="utf-8")
    monkeypatch.setattr(datasets, "PREPARED_ROOT", tmp_path / "prepared")

    result = datasets.inspect_dataset(str(source))

    assert result["status"] == "ready", result
    assert result["layout"]["mode"] == "roboflow"
    assert result["splits"] == {"train": 4, "val": 2, "test": 2}
    assert result["classes"] == ["drone"]
    assert result["source_unchanged"] is True


def test_train_only_dataset_gets_configurable_cached_splits(tmp_path, monkeypatch):
    source = tmp_path / "train-only"
    for index in range(10):
        image = source / "train" / "images" / f"image-{index}.jpg"
        _image(image, value=index * 10)
        label = source / "train" / "labels" / f"image-{index}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    monkeypatch.setattr(datasets, "PREPARED_ROOT", tmp_path / "prepared")

    default = datasets.inspect_dataset(str(source))
    no_test = datasets.inspect_dataset(
        str(source),
        options={"split": {"create_val": True, "create_test": False, "ratios": {"train": 0.9, "val": 0.1, "test": 0}, "seed": 7}},
    )

    assert default["status"] == "ready", default
    assert default["splits"] == {"train": 8, "val": 1, "test": 1}
    assert default["split_policy"]["generated"] == ["val", "test"]
    assert no_test["status"] == "ready", no_test
    assert no_test["splits"] == {"train": 9, "val": 1, "test": 0}
    assert no_test["split_policy"]["seed"] == 7


def test_yolo_boundary_rounding_uses_ultralytics_tolerance_and_clipping(tmp_path, monkeypatch):
    source = tmp_path / "boundary"
    image_path = source / "train" / "images" / "edge.jpg"
    _image(image_path, width=1280, height=720)
    label_path = source / "train" / "labels" / "edge.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.592515625 0.36675 0.64565625 0.7335138888888889\n", encoding="utf-8")
    monkeypatch.setattr(datasets, "PREPARED_ROOT", tmp_path / "prepared")

    result = datasets.inspect_dataset(
        str(source),
        options={"split": {"create_val": False, "create_test": False, "ratios": {"train": 1, "val": 0, "test": 0}, "seed": 42}},
    )

    assert result["status"] == "ready", result
    assert result["annotation_repairs"]["clipped_boxes"] == 1
    prepared_label = next((tmp_path / "prepared").rglob("labels/train/*.txt"))
    values = [float(value) for value in prepared_label.read_text(encoding="utf-8").split()]
    assert 0 <= values[2] - values[4] / 2
    assert values[2] + values[4] / 2 <= 1


def test_yolo_substantially_out_of_bounds_coordinates_are_still_blocked(tmp_path, monkeypatch):
    source = tmp_path / "invalid-boundary"
    _image(source / "train" / "images" / "bad.jpg")
    label = source / "train" / "labels" / "bad.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("0 0.5 0.5 1.2 0.2\n", encoding="utf-8")
    monkeypatch.setattr(datasets, "PREPARED_ROOT", tmp_path / "prepared")

    result = datasets.inspect_dataset(str(source), options={"split": {"create_val": False, "create_test": False, "ratios": {"train": 1, "val": 0, "test": 0}, "seed": 42}})

    assert result["status"] == "blocked"
    assert "normalized coordinates" in result["message"]


def test_manual_mapping_recovers_nonstandard_yolo_paths(tmp_path, monkeypatch):
    source = tmp_path / "manual"
    for index in range(4):
        image = source / "pictures" / f"image-{index}.jpg"
        _image(image, value=index * 20)
        label = source / "annotations" / f"image-{index}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    monkeypatch.setattr(datasets, "PREPARED_ROOT", tmp_path / "prepared")

    result = datasets.inspect_dataset(
        str(source),
        options={"layout": "manual", "classes": ["drone"], "splits": {"train": {"images": "pictures", "labels": "annotations"}}, "split": {"create_val": True, "create_test": False, "ratios": {"train": 0.67, "val": 0.33, "test": 0}, "seed": 42}},
    )

    assert result["status"] == "ready", result
    assert result["layout"]["mode"] == "manual"
    assert result["classes"] == ["drone"]
    assert result["splits"]["val"] == 1


def test_nested_dataset_paths_are_relative_to_selected_root_for_manual_mapping(tmp_path, monkeypatch):
    source = tmp_path / "archive-wrapper"
    nested = source / "drone-v1"
    for index in range(4):
        image = nested / "train" / "images" / f"image-{index}.jpg"
        _image(image, value=index * 20)
        label = nested / "train" / "labels" / f"image-{index}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (nested / "data.yaml").write_text("names: [drone]\ntrain: train/images\n", encoding="utf-8")
    monkeypatch.setattr(datasets, "PREPARED_ROOT", tmp_path / "prepared")

    inspected = datasets.inspect_dataset(str(source))
    assert inspected["status"] == "ready", inspected
    assert inspected["layout"]["splits"]["train"]["images"] == "drone-v1/train/images"

    manual = datasets.inspect_dataset(
        str(source),
        options={
            "layout": "manual",
            "splits": {"train": {"images": "drone-v1/train/images", "labels": "drone-v1/train/labels"}},
            "split": {"create_val": False, "create_test": False, "ratios": {"train": 1, "val": 0, "test": 0}, "seed": 42},
        },
    )
    assert manual["status"] == "ready", manual

    blocked = datasets.inspect_dataset(
        str(source),
        options={"layout": "manual", "splits": {"train": {"images": "missing/images", "labels": "missing/labels"}}},
    )
    assert blocked["status"] == "blocked"
    assert blocked["layout"]["mode"] == "manual"
    assert blocked["layout"]["splits"]["train"]["images"] == "missing/images"

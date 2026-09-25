from typing import Any, Dict, List

from ultralytics.cfg import DEFAULT_CFG_DICT, smart_value


TRAIN_GROUPS = {
    "Optimizer": [
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "nbs",
        "cos_lr",
    ],
    "Augmentation": [
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "bgr",
        "mosaic",
        "mixup",
        "close_mosaic",
        "multi_scale",
    ],
    "Training behavior": [
        "seed",
        "deterministic",
        "single_cls",
        "rect",
        "amp",
        "fraction",
        "cache",
        "freeze",
        "profile",
    ],
    "Validation": [
        "save_period",
        "val_period",
        "plots",
        "save_json",
        "save_hybrid",
        "conf",
        "iou",
        "max_det",
        "half",
    ],
}

PREDICT_GROUPS = {
    "Inference": [
        "half",
        "max_det",
        "batch",
        "vid_stride",
        "stream_buffer",
        "visualize",
        "augment",
        "agnostic_nms",
        "classes",
    ],
    "Output": [
        "save_frames",
        "save_txt",
        "save_conf",
        "save_crop",
        "show_labels",
        "show_conf",
        "show_boxes",
        "line_width",
    ],
}


def default_cfg_dict() -> Dict[str, Any]:
    cfg = dict(DEFAULT_CFG_DICT)
    cfg["amp"] = False
    return cfg


def build_grouped_defaults(mode: str) -> Dict[str, Dict[str, Any]]:
    cfg = default_cfg_dict()
    groups = TRAIN_GROUPS if mode == "train" else PREDICT_GROUPS
    grouped: Dict[str, Dict[str, Any]] = {}
    for group_name, keys in groups.items():
        group_items = {}
        for key in keys:
            if key in cfg:
                group_items[key] = cfg[key]
        if group_items:
            grouped[group_name] = group_items
    return grouped


def coerce_value(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return bool(smart_value(value))
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(value, str):
        return smart_value(value)
    return value


def coerce_dict(values: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    coerced = {}
    for key, value in values.items():
        default = defaults.get(key)
        coerced[key] = coerce_value(value, default)
    return coerced

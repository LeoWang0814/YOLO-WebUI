"""Form definitions and coercion for the server-rendered workbench."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple

from core.args_schema import build_grouped_defaults, coerce_dict


MANAGED_FIELDS = {
    "train": {"task", "mode", "data", "model", "project", "name", "exist_ok", "epochs", "patience", "batch", "imgsz", "device", "workers", "verbose"},
    "predict": {"task", "mode", "model", "source", "project", "name", "exist_ok", "imgsz", "conf", "iou", "device", "save"},
}


def expert_groups(mode: str) -> List[Tuple[str, List[Tuple[str, Any]]]]:
    seen = set()
    groups = []
    for group_name, values in build_grouped_defaults(mode).items():
        fields = []
        for key, default in values.items():
            if key in seen or key in MANAGED_FIELDS[mode]:
                continue
            seen.add(key)
            fields.append((key, default))
        if fields:
            groups.append((group_name, fields))
    return groups


def form_control(default: Any) -> str:
    if isinstance(default, bool):
        return "checkbox"
    if isinstance(default, (int, float)):
        return "number"
    return "text"


def _last_value(form: Mapping[str, Any], key: str) -> Any:
    getlist = getattr(form, "getlist", None)
    if getlist:
        values = getlist(key)
        return values[-1] if values else None
    return form.get(key)


def expert_values(mode: str, form: Mapping[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    raw: Dict[str, Any] = {}
    for _, fields in expert_groups(mode):
        for key, default in fields:
            field_name = f"expert__{key}"
            value = _last_value(form, field_name)
            if value is not None:
                defaults[key] = default
                raw[key] = value
    return {key: value for key, value in coerce_dict(raw, defaults).items() if value is not None}


def required_text(form: Mapping[str, Any], key: str, label: str) -> str:
    value = str(form.get(key) or "").strip()
    if not value:
        raise ValueError(f"{label} is required.")
    return value


def integer(form: Mapping[str, Any], key: str, label: str, minimum: int = 0, default: int | None = None) -> int:
    raw = form.get(key)
    if raw in (None, "") and default is not None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return value


def number(form: Mapping[str, Any], key: str, label: str, minimum: float = 0.0, default: float | None = None) -> float:
    raw = form.get(key)
    if raw in (None, "") and default is not None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return value


def form_list(form: Mapping[str, Any], key: str) -> List[str]:
    getlist = getattr(form, "getlist", None)
    if getlist:
        return [str(value) for value in getlist(key) if str(value).strip()]
    value = form.get(key)
    return [str(value)] if value else []

"""GPU inventory and compatibility checks for the local PyTorch runtime."""

from __future__ import annotations

import platform
import re
import subprocess
import ctypes
from pathlib import Path
from typing import Dict, List, Tuple

import torch


CUDA_DRIVER_ERRORS = {
    0: "CUDA_SUCCESS",
    3: "CUDA_ERROR_INITIALIZATION_ERROR",
    100: "CUDA_ERROR_NO_DEVICE",
    803: "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH",
    999: "CUDA_ERROR_UNKNOWN",
}


def cuda_driver_probe() -> Dict[str, object]:
    """Call the driver API directly so NVML-only visibility is distinguishable."""
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        init = driver.cuInit
        init.argtypes = [ctypes.c_uint]
        init.restype = ctypes.c_int
        code = int(init(0))
    except (AttributeError, OSError) as exc:
        return {"code": None, "name": "CUDA_DRIVER_LIBRARY_UNAVAILABLE", "detail": str(exc)}
    return {"code": code, "name": CUDA_DRIVER_ERRORS.get(code, f"CUDA_ERROR_{code}"), "detail": ""}


def _device_nodes() -> List[str]:
    return sorted(path.name for path in Path("/dev").glob("nvidia*") if path.name != "nvidia-caps")


def _query_nvidia_smi() -> List[Dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        index, name, memory_total = parts
        gpus.append({"index": index, "name": name, "memory_gb": f"{float(memory_total) / 1024:.1f}"})
    return gpus


def _query_nvidia_smi_summary() -> Dict[str, str]:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    match = re.search(r"Driver Version:\s*([0-9.]+).*?CUDA Version:\s*([0-9.]+)", result.stdout, re.DOTALL)
    if not match:
        return {}
    return {"driver_version": match.group(1), "driver_cuda": match.group(2)}


def _compiled_cuda_architectures() -> List[str]:
    try:
        return sorted(set(torch.cuda.get_arch_list())) if torch.cuda.is_available() else []
    except (AttributeError, RuntimeError):
        return []


def _architecture_supported(capability: str, architectures: List[str]) -> bool:
    return capability in architectures or capability.replace("sm_", "compute_") in architectures


def cuda_device_report() -> Dict[str, object]:
    """Describe CUDA devices and whether this wheel contains kernels for them."""
    driver_probe = cuda_driver_probe()
    cuda_available = torch.cuda.is_available()
    architectures = _compiled_cuda_architectures()
    gpus: List[Dict[str, object]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            capability = f"sm_{properties.major}{properties.minor}"
            gpus.append(
                {
                    "index": str(index),
                    "name": properties.name,
                    "memory_gb": f"{properties.total_memory / (1024 ** 3):.1f}",
                    "capability": capability,
                    "compatible": _architecture_supported(capability, architectures),
                }
            )
    else:
        gpus = [{**gpu, "capability": "", "compatible": False} for gpu in _query_nvidia_smi()]
    compatible = [gpu for gpu in gpus if gpu["compatible"]]
    incompatible = [gpu for gpu in gpus if not gpu["compatible"]]
    return {
        "cuda_available": cuda_available,
        "architectures": architectures,
        "gpus": gpus,
        "compatible": compatible,
        "incompatible": incompatible,
        "driver_probe": driver_probe,
        "device_nodes": _device_nodes(),
    }


def compatible_cuda_device_ids() -> List[str]:
    return [str(gpu["index"]) for gpu in cuda_device_report()["compatible"]]


def _cuda_runtime_error(report: Dict[str, object]) -> str:
    gpus = report["gpus"]
    if not gpus:
        return ""
    if not report["cuda_available"]:
        probe = report["driver_probe"]
        if probe["code"] == 999:
            return (
                "nvidia-smi can see the GPU, but CUDA driver initialization returned CUDA_ERROR_UNKNOWN (999). "
                "Recreate the NVIDIA-enabled container or repair/reload the host nvidia_uvm runtime before using GPU training."
            )
        if probe["code"] == 803:
            return "The loaded NVIDIA driver and CUDA compatibility library do not match. Use the host-matched driver runtime."
        return "NVIDIA GPU detected, but PyTorch could not initialize CUDA. Install the matching CUDA runtime or choose CPU."
    if report["incompatible"]:
        unsupported = ", ".join(f"GPU {gpu['index']} ({gpu['capability']})" for gpu in report["incompatible"])
        architectures = ", ".join(report["architectures"]) or "none"
        return (
            f"Installed PyTorch has no kernels for {unsupported}; it supports {architectures}. "
            "Install requirements-cuda128.txt for RTX 50-series / Blackwell GPUs."
        )
    return ""


def cuda_runtime_error() -> str:
    return _cuda_runtime_error(cuda_device_report())


def _format_gpus(report: Dict[str, object]) -> Tuple[List[Dict[str, object]], str]:
    gpus = report["gpus"]
    return gpus, "torch" if report["cuda_available"] else ("nvidia-smi" if gpus else "none")


def get_system_status() -> Dict[str, object]:
    report = cuda_device_report()
    gpus, gpu_source = _format_gpus(report)
    smi_summary = _query_nvidia_smi_summary()
    runtime_error = _cuda_runtime_error(report)
    if runtime_error:
        cuda_state = "incompatible"
        cuda_note = "CUDA runtime unavailable." if not report["cuda_available"] else "PyTorch GPU architecture mismatch."
        cuda_hint = runtime_error
    elif report["cuda_available"]:
        cuda_state = "available"
        cuda_note = "OK"
        cuda_hint = ""
    else:
        cuda_state = "unavailable"
        cuda_note = "No CUDA runtime detected."
        cuda_hint = ""
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda or "N/A",
        "cuda": cuda_state,
        "cuda_note": cuda_note,
        "cuda_hint": cuda_hint,
        "gpu_count": len(gpus),
        "compatible_gpu_count": len(report["compatible"]),
        "gpu_list": gpus,
        "gpu_source": gpu_source,
        "compiled_architectures": report["architectures"],
        "cuda_driver_probe": report["driver_probe"],
        "cuda_device_nodes": report["device_nodes"],
        "driver_version": smi_summary.get("driver_version", ""),
        "driver_cuda": smi_summary.get("driver_cuda", ""),
        "platform": platform.platform(),
    }

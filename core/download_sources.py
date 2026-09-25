"""Select model delivery sources from the server's public egress address."""

from __future__ import annotations

import ipaddress
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple, Union

import requests


ROOT = Path(__file__).resolve().parents[1]
IP_DATA_ROOT = ROOT / "tools" / "ip_data"
_EGRESS_IP_URL = "https://api.ipify.org"
_CACHE_SECONDS = 15 * 60
_cache_lock = Lock()
_cached_source: Optional[Tuple[float, str]] = None
Network = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


@lru_cache(maxsize=2)
def _china_networks(version: int) -> Tuple[Network, ...]:
    filename = "cn_ipv4_aggregated.txt" if version == 4 else "cn_ipv6_aggregated.txt"
    path = IP_DATA_ROOT / filename
    networks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cidr = line.strip()
        if not cidr or cidr.startswith("#"):
            continue
        network = ipaddress.ip_network(cidr, strict=True)
        if network.version == version:
            networks.append(network)
    return tuple(networks)


def is_mainland_china_ip(value: str) -> bool:
    """Return whether an IPv4 or IPv6 address is in a bundled mainland CN CIDR."""
    try:
        address = ipaddress.ip_address(value.strip())
    except ValueError:
        return False
    return any(address in network for network in _china_networks(address.version))


def _public_egress_ip() -> Optional[str]:
    try:
        with requests.get(
            _EGRESS_IP_URL,
            headers={"User-Agent": "yolov10-webui/1.0"},
            timeout=(2, 3),
        ) as response:
            response.raise_for_status()
            address = ipaddress.ip_address(response.text.strip())
    except (ValueError, requests.RequestException):
        return None
    return str(address) if address.is_global else None


def preferred_model_source() -> str:
    """Choose Nanoberry for mainland-China egress and GitHub everywhere else.

    A failed egress lookup uses Nanoberry first. GitHub remains a download fallback,
    so an unavailable lookup service cannot prevent a model from being downloaded.
    """
    global _cached_source
    now = time.monotonic()
    with _cache_lock:
        if _cached_source and now - _cached_source[0] < _CACHE_SECONDS:
            return _cached_source[1]

    address = _public_egress_ip()
    source = "nanoberry" if address is None or is_mainland_china_ip(address) else "github"
    with _cache_lock:
        _cached_source = (now, source)
    return source

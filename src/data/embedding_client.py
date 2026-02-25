"""TEI (Text Embeddings Inference) REST API client (KIK-420).

Provides embedding generation via Hugging Face TEI Docker service.
Graceful degradation: returns None when TEI is unavailable.
"""

import logging
import os
import threading
import time

import requests

logger = logging.getLogger(__name__)

TEI_URL = os.environ.get("TEI_URL", "http://localhost:8081")

_available: bool | None = None
_available_checked_at: float = 0.0
_AVAILABILITY_TTL = 30.0  # re-check every 30s
_state_lock = threading.Lock()


def is_available() -> bool:
    """Check if TEI service is reachable (result cached for 30s).

    Why: TEI availability check involves network I/O; caching avoids
         repeated round-trips within the TTL window.
    How: Lock protects the global _available/_available_checked_at
         from concurrent session threads in Streamlit.
    """
    global _available, _available_checked_at
    with _state_lock:
        now = time.time()
        if _available is not None and (now - _available_checked_at) < _AVAILABILITY_TTL:
            return _available
        try:
            resp = requests.get(f"{TEI_URL}/health", timeout=3)
            _available = resp.status_code == 200
        except Exception as exc:
            logger.debug("TEI health check failed: %s", exc)
            _available = False
        _available_checked_at = now
        return _available


def get_embedding(text: str) -> list[float] | None:
    """Get embedding vector from TEI. Returns None on failure."""
    if not text:
        return None
    try:
        resp = requests.post(
            f"{TEI_URL}/embed",
            json={"inputs": text},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
    except Exception as exc:
        logger.debug("TEI embedding request failed: %s", exc)
    return None


def reset_cache() -> None:
    """Reset availability cache (for testing)."""
    global _available, _available_checked_at
    with _state_lock:
        _available = None
        _available_checked_at = 0.0

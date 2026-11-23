"""Tiny shared HTTP helper for the REST-only LLM providers (urllib, no SDKs)."""
from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .provider import LLMProviderError


def ssl_context() -> Optional[ssl.SSLContext]:
    """SSL context backed by certifi's CA bundle (stock macOS python.org builds
    ship without system CA certificates); falls back to the platform default."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        try:
            return ssl.create_default_context()
        except Exception:
            return None


class HTTPStatusError(LLMProviderError):
    """Non-2xx provider response. ``status`` and the parsed/raw body are kept so
    callers can branch (e.g. retry without an unsupported parameter)."""

    def __init__(self, service: str, status: int, detail: str):
        self.service, self.status, self.detail = service, status, detail
        super().__init__(f"{service} returned HTTP {status}: {_short_detail(detail)}")


def _short_detail(detail: str) -> str:
    """Prefer the provider's human-readable ``error.message`` over a JSON blob."""
    try:
        data = json.loads(detail)
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
    except Exception:
        pass
    return detail.strip()[:600]


def request_json(service: str, url: str, *, headers: Dict[str, str],
                 payload: Optional[Dict[str, Any]] = None, timeout: float = 90) -> Dict[str, Any]:
    """POST ``payload`` as JSON (or GET when it is None) and return the JSON body."""
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=body, method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise HTTPStatusError(service, exc.code, detail) from exc
    except urllib.error.URLError as exc:
        raise LLMProviderError(f"{service} network error: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise LLMProviderError(f"{service} returned invalid JSON: {exc}") from exc

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from ._http import HTTPStatusError, request_json
from .credentials import new_local_interaction_id
from .provider import LLMProvider, LLMProviderError, LLMResponse

logger = logging.getLogger(__name__)

_ENV_KEY = "ANTHROPIC_API_KEY"
_BASE_URL = "https://api.anthropic.com/v1"
_API_VERSION = "2026-10-21"
_DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")
_MAX_TOKENS = 16000
_TIMEOUT = 90   # stay near the reverse proxy's patience; one retry at most
# Styling a map is a small structured task: low effort keeps the UI responsive.
_EFFORT = os.environ.get("ANTHROPIC_EFFORT", "low").strip().lower()
_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
# Models whose safety classifiers can decline a request and that support the
# server-side fallback, which re-runs a declined request on another model.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-fable-5-2")


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) provider over the Messages REST API.

    Raw REST by project convention (no SDKs in this package). Stateless, like
    the OpenAI provider: a synthetic ``local-…`` interaction id is returned.
    Sampling parameters are not sent (this API version rejects them); the
    ``temperature`` argument is accepted for interface compatibility only.
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get(_ENV_KEY)
        if not self._api_key:
            raise LLMProviderError(
                f"No Anthropic API key: add your own key in the UI or set {_ENV_KEY}."
            )
        self._model = model or _DEFAULT_MODEL

    def _headers(self, beta: bool) -> Dict[str, str]:
        headers = {"x-api-key": self._api_key, "anthropic-version": _API_VERSION}
        if beta:
            headers["anthropic-beta"] = _FALLBACK_BETA
        return headers

    def generate_response(
        self,
        prompt: str,
        *,
        previous_interaction_id: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self._model,
            "max_tokens": _MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_instruction:
            payload["system"] = system_instruction
        if _EFFORT in _EFFORTS:
            payload["output_config"] = {"effort": _EFFORT}
        use_fallback = self._model.startswith(_FALLBACK_MODELS)
        if use_fallback:
            payload["fallbacks"] = "default"

        try:
            data = request_json("Anthropic", f"{_BASE_URL}/messages",
                                headers=self._headers(use_fallback), payload=payload, timeout=_TIMEOUT)
        except HTTPStatusError as exc:
            # A model that does not take the optional extras: retry once with the bare request.
            detail = exc.detail.lower()
            if exc.status == 400 and any(k in detail for k in ("fallback", "output_config", "effort", "beta")):
                payload.pop("fallbacks", None)
                payload.pop("output_config", None)
                data = request_json("Anthropic", f"{_BASE_URL}/messages",
                                    headers=self._headers(False), payload=payload, timeout=_TIMEOUT)
            else:
                raise

        if data.get("stop_reason") == "refusal":
            details = data.get("stop_details") if isinstance(data.get("stop_details"), dict) else {}
            raise LLMProviderError(
                "Claude declined this request"
                + (f" ({details.get('category')})" if details.get("category") else "") + "."
            )
        text = "\n".join(
            b["text"].strip() for b in (data.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str) and b["text"].strip()
        )
        if not text:
            if data.get("stop_reason") == "max_tokens":
                raise LLMProviderError("Claude ran out of output tokens before answering.")
            raise LLMProviderError("Could not parse the Anthropic response.")
        return LLMResponse(text=text, interaction_id=new_local_interaction_id(), raw=data)


def list_models(api_key: str) -> List[str]:
    """Model ids visible to ``api_key`` (also validates the key)."""
    data = request_json("Anthropic", f"{_BASE_URL}/models?limit=100",
                        headers={"x-api-key": api_key, "anthropic-version": _API_VERSION}, timeout=30)
    return [str(m["id"]) for m in (data.get("data") or []) if isinstance(m, dict) and m.get("id")]

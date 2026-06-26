from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .provider import LLMProvider, LLMProviderError, LLMResponse

logger = logging.getLogger(__name__)

_ENV_KEY = "GEMINI_API_KEY"
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
_DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")


def _ssl_context() -> Optional[ssl.SSLContext]:
    """SSL context backed by certifi's CA bundle.

    Stock macOS python.org builds ship without system CA certificates, so
    urllib's default HTTPS verification fails against Google's API. Prefer
    certifi (a hard dependency of common stacks) and fall back to the platform
    default if it is unavailable.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        try:
            return ssl.create_default_context()
        except Exception:
            return None


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the Interactions REST API.

    Uses the stateful Interactions API so callers can continue a conversation
    by passing `previous_interaction_id`.
    """

    def __init__(self, model: str = _DEFAULT_MODEL):
        self._api_key = os.environ.get(_ENV_KEY)
        if not self._api_key:
            raise LLMProviderError(
                f"Environment variable {_ENV_KEY} is not set. "
                "Obtain a key at https://aistudio.google.com/apikey"
            )
        self._model = model

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
            "input": prompt,
        }

        if previous_interaction_id:
            payload["previous_interaction_id"] = previous_interaction_id

        if system_instruction:
            payload["system_instruction"] = system_instruction

        if temperature is not None:
            payload["generation_config"] = {
                "temperature": float(temperature)
            }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _GEMINI_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=90, context=_ssl_context()) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            logger.error("Gemini Interactions HTTP %s: %s", exc.code, detail)
            raise LLMProviderError(
                f"Gemini Interactions API returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            logger.error("Gemini Interactions network error: %s", exc.reason)
            raise LLMProviderError(
                f"Gemini Interactions network error: {exc.reason}"
            ) from exc
        except json.JSONDecodeError as exc:
            logger.error("Gemini Interactions returned invalid JSON")
            raise LLMProviderError(
                f"Gemini Interactions returned invalid JSON: {exc}"
            ) from exc

        interaction_id = data.get("id")
        text = self._extract_text(data)

        return LLMResponse(
            text=text,
            interaction_id=interaction_id,
            raw=data,
        )

    def _extract_text(self, data: Dict[str, Any]) -> str:
        """Best-effort extraction of text from an Interaction response.

        The Interactions API returns assistant text inside
        ``steps[].content[].text`` (the step whose type is ``model_output``;
        ``thought`` steps carry no visible text). Older/alternate shapes used a
        top-level ``outputs[]`` array, which we still accept as a fallback.
        """
        text_chunks: List[str] = []

        # Primary shape: steps[].content[].text
        for step in data.get("steps") or []:
            if not isinstance(step, dict):
                continue
            for item in step.get("content") or []:
                if isinstance(item, dict):
                    item_text = item.get("text")
                    if isinstance(item_text, str) and item_text.strip():
                        text_chunks.append(item_text.strip())

        # Fallback shape: outputs[].text / outputs[].parts[].text
        if not text_chunks:
            for output in data.get("outputs") or []:
                if isinstance(output, dict):
                    direct_text = output.get("text")
                    if isinstance(direct_text, str) and direct_text.strip():
                        text_chunks.append(direct_text.strip())
                    for part in output.get("parts") or []:
                        if isinstance(part, dict):
                            part_text = part.get("text")
                            if isinstance(part_text, str) and part_text.strip():
                                text_chunks.append(part_text.strip())

        if text_chunks:
            return "\n".join(text_chunks)

        logger.error("Unexpected Gemini Interactions response shape: %s", data)
        raise LLMProviderError(
            "Could not parse Gemini Interactions response text."
        )
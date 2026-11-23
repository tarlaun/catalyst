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


from ._http import ssl_context as _ssl_context  # noqa: E402  (kept under its old name)
from .credentials import LOCAL_INTERACTION_PREFIX  # noqa: E402


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the Interactions REST API.

    Uses the stateful Interactions API so callers can continue a conversation
    by passing `previous_interaction_id`.
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get(_ENV_KEY)
        if not self._api_key:
            raise LLMProviderError(
                f"No Gemini API key: add your own key in the UI or set {_ENV_KEY}. "
                "Obtain a key at https://aistudio.google.com/apikey"
            )
        self._model = model or _DEFAULT_MODEL

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

        # Ids minted by the stateless providers mean nothing to Gemini.
        if previous_interaction_id and not previous_interaction_id.startswith(LOCAL_INTERACTION_PREFIX):
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


def list_models(api_key: str) -> List[str]:
    """Text-generation model ids visible to ``api_key`` (also validates the key)."""
    from ._http import request_json
    data = request_json(
        "Gemini", "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
        headers={"x-goog-api-key": api_key}, timeout=30,
    )
    skip = ("embedding", "aqa", "imagen", "veo", "tts", "audio", "image", "live", "robotics", "computer-use")
    ids = []
    for m in data.get("models") or []:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").removeprefix("models/")
        if (name.startswith("gemini") and "generateContent" in (m.get("supportedGenerationMethods") or [])
                and not any(k in name for k in skip)):
            ids.append(name)
    return sorted(set(ids), reverse=True)

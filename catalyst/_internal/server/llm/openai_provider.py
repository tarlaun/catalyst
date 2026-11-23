from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from ._http import HTTPStatusError, request_json
from .credentials import new_local_interaction_id
from .provider import LLMProvider, LLMProviderError, LLMResponse

logger = logging.getLogger(__name__)

_ENV_KEY = "OPENAI_API_KEY"
_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.1")


class OpenAIProvider(LLMProvider):
    """OpenAI provider over the Chat Completions REST API.

    Stateless: every Catalyst prompt already carries the dataset, the current
    style and the attribute hints, so follow-up turns do not need server-side
    history. A synthetic ``local-…`` interaction id is returned so the client
    keeps treating the exchange as one conversation.
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get(_ENV_KEY)
        if not self._api_key:
            raise LLMProviderError(
                f"No OpenAI API key: add your own key in the UI or set {_ENV_KEY}."
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
        messages: List[Dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {"model": self._model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = float(temperature)

        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            data = request_json("OpenAI", f"{_BASE_URL}/chat/completions", headers=headers, payload=payload)
        except HTTPStatusError as exc:
            # Reasoning models only accept the default temperature: retry once without it.
            if exc.status == 400 and "temperature" in payload and "temperature" in exc.detail.lower():
                payload.pop("temperature")
                data = request_json("OpenAI", f"{_BASE_URL}/chat/completions", headers=headers, payload=payload)
            else:
                raise

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("Could not parse the OpenAI response.") from exc
        text = message.get("content") if isinstance(message, dict) else None
        if isinstance(text, list):  # content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        if not isinstance(text, str) or not text.strip():
            refusal = message.get("refusal") if isinstance(message, dict) else None
            raise LLMProviderError(f"OpenAI returned no text{': ' + str(refusal) if refusal else '.'}")
        return LLMResponse(text=text.strip(), interaction_id=new_local_interaction_id(), raw=data)


def list_models(api_key: str) -> List[str]:
    """Chat-capable model ids visible to ``api_key`` (also validates the key)."""
    data = request_json("OpenAI", f"{_BASE_URL}/models",
                        headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    skip = ("embedding", "audio", "tts", "whisper", "transcribe", "realtime", "image", "dall-e",
            "moderation", "search", "instruct", "codex", "computer-use", "sora", "deep-research", "-pro")
    ids = []
    for m in data.get("data") or []:
        mid = str(m.get("id") or "") if isinstance(m, dict) else ""
        if mid.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")) and not any(s in mid for s in skip):
            ids.append(mid)
    return sorted(set(ids), reverse=True)

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

from .provider import LLMProvider, LLMProviderError, LLMResponse

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"
_OLLAMA_URL = "http://localhost:11434/api/generate"


class OllamaProvider(LLMProvider):
    """Ollama provider using the local REST API.

    This provider is stateless. It accepts the same interface as Gemini for
    compatibility, but `previous_interaction_id` is ignored and the returned
    `interaction_id` is always None.
    """

    def __init__(self, model: Optional[str] = None):
        self._model = model or os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)

    def generate_response(
        self,
        prompt: str,
        *,
        previous_interaction_id: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        composed_prompt = self._compose_prompt(
            prompt=prompt,
            system_instruction=system_instruction,
        )

        payload = {
            "model": self._model,
            "prompt": composed_prompt,
            "stream": False,
        }

        if temperature is not None:
            payload["options"] = {"temperature": float(temperature)}

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _OLLAMA_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            logger.error("Ollama HTTP %s: %s", exc.code, detail)
            raise LLMProviderError(
                f"Ollama API returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            logger.error("Ollama network error: %s", exc.reason)
            raise LLMProviderError(
                f"Ollama network error: {exc.reason}"
            ) from exc

        try:
            text = data["response"]
        except (KeyError, TypeError) as exc:
            logger.error("Unexpected Ollama response shape: %s", data)
            raise LLMProviderError(
                f"Could not parse Ollama response: {exc}"
            ) from exc

        return LLMResponse(
            text=text,
            interaction_id=None,
            raw=data,
        )

    @staticmethod
    def _compose_prompt(prompt: str, system_instruction: Optional[str]) -> str:
        if not system_instruction:
            return prompt
        return (
            "System instruction:\n"
            f"{system_instruction.strip()}\n\n"
            "User request:\n"
            f"{prompt}"
        )
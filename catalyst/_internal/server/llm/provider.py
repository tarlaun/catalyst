from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMResponse:
    """Normalized provider response.

    Attributes:
        text:
            The provider's plain-text output.
        interaction_id:
            Conversation/session identifier when supported by the provider.
            For Gemini Interactions API this is the interaction `id`.
            For stateless providers this may be None.
        raw:
            Optional raw provider payload for debugging.
    """

    text: str
    interaction_id: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract base interface for LLM providers.

    Every concrete provider (Gemini, Ollama, etc.) implements the same method
    so callers remain provider-agnostic.
    """

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        *,
        previous_interaction_id: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Send *prompt* to the underlying LLM and return a normalized response.

        Args:
            prompt:
                User input or task prompt.
            previous_interaction_id:
                Optional prior conversation/interaction id. Providers that do
                not support server-side conversation state may ignore it.
            system_instruction:
                Optional system prompt / instruction to apply for this turn.
            temperature:
                Optional generation temperature.

        Raises:
            LLMProviderError:
                On any communication or API failure.
        """
        raise NotImplementedError


class LLMProviderError(Exception):
    """Raised when an LLM provider fails to produce a response."""
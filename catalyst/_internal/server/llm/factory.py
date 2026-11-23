import logging
import os

from .credentials import current_credentials, note_used
from .provider import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)

_DEFAULT_PROVIDER = "gemini"

# Registry of provider name -> callable that returns an LLMProvider instance.
# Each entry is a zero-arg factory so that imports and API-key validation are
# deferred until the provider is actually requested.
_PROVIDERS = {
    "gemini": lambda: _make_gemini(),
    "openai": lambda: _make_openai(),
    "anthropic": lambda: _make_anthropic(),
    "ollama": lambda: _make_ollama(),
    "fallback": lambda: _make_fallback(),
}


def _make_openai() -> LLMProvider:
    from .openai_provider import OpenAIProvider
    return OpenAIProvider()


def _make_anthropic() -> LLMProvider:
    from .anthropic_provider import AnthropicProvider
    return AnthropicProvider()


def _make_for_credentials(creds) -> LLMProvider:
    """Provider built from a visitor's own key (see ``credentials.py``)."""
    if creds.provider == "gemini":
        from .gemini_provider import GeminiProvider
        return GeminiProvider(model=creds.model, api_key=creds.api_key)
    if creds.provider == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(model=creds.model, api_key=creds.api_key)
    if creds.provider == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=creds.model, api_key=creds.api_key)
    raise LLMProviderError(f"Unsupported bring-your-own-key provider '{creds.provider}'.")


def list_models_for(provider: str, api_key: str):
    """Model ids the key can use; raises LLMProviderError when the key is rejected."""
    if provider == "gemini":
        from .gemini_provider import list_models
    elif provider == "openai":
        from .openai_provider import list_models
    elif provider == "anthropic":
        from .anthropic_provider import list_models
    else:
        raise LLMProviderError(f"Unsupported provider '{provider}'.")
    return list_models(api_key)


def _make_gemini() -> LLMProvider:
    from .gemini_provider import GeminiProvider
    return GeminiProvider()


def _make_ollama() -> LLMProvider:
    from .ollama_provider import OllamaProvider
    return OllamaProvider()


class _FallbackProvider(LLMProvider):
    """Tries an ordered list of providers; uses the first that succeeds.

    Construction OR generation failure (e.g. missing/expired Gemini key, network
    error) falls through to the next provider. Lets a deployment prefer Gemini
    while staying up on a local Ollama when Gemini is unavailable.
    """

    def __init__(self, builders, names):
        self._builders = list(builders)
        self._names = list(names)
        self._cache = {}

    def generate_response(self, prompt, *, previous_interaction_id=None,
                          system_instruction=None, temperature=None):
        last_err = None
        for i, build in enumerate(self._builders):
            name = self._names[i]
            try:
                prov = self._cache.get(name)
                if prov is None:
                    prov = build()
                    self._cache[name] = prov
                # Only the first (stateful) provider can use a prior interaction
                # id; stateless fallbacks (Ollama) ignore it anyway.
                pid = previous_interaction_id if i == 0 else None
                return prov.generate_response(
                    prompt,
                    previous_interaction_id=pid,
                    system_instruction=system_instruction,
                    temperature=temperature,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning("LLM provider '%s' failed (%s); trying next.", name, exc)
                self._cache.pop(name, None)
                continue
        raise LLMProviderError(f"All LLM providers failed. Last error: {last_err}")


def _make_fallback() -> LLMProvider:
    """Build an ordered fallback chain from ``LLM_FALLBACK_ORDER`` (default
    ``gemini,ollama``)."""
    order = [n.strip().lower() for n in
             os.environ.get("LLM_FALLBACK_ORDER", "gemini,ollama").split(",") if n.strip()]
    builders, names = [], []
    for n in order:
        b = _PROVIDERS.get(n)
        if b is not None and n != "fallback":
            builders.append(b)
            names.append(n)
    if not builders:
        builders, names = [_make_gemini], ["gemini"]
    return _FallbackProvider(builders, names)


def require_user_key() -> bool:
    """``CATALYST_REQUIRE_USER_KEY=1``: never spend the deployment's own LLM key
    on visitors; they must bring their own (heuristics are used otherwise)."""
    return os.environ.get("CATALYST_REQUIRE_USER_KEY", "0").strip().lower() in ("1", "true", "yes", "on")


class LLMFactory:
    """Instantiate an :class:`LLMProvider` by name.

    Supported names (case-insensitive):
        * ``"gemini"`` — Google Gemini Interactions API
        * ``"openai"`` — OpenAI Chat Completions API
        * ``"anthropic"`` — Anthropic Messages API
        * ``"ollama"`` — Local Ollama

    When the current request carries a visitor's own API key, that provider
    and key are used regardless of *name*.

    Future providers can be added by registering a builder in `_PROVIDERS`.
    """

    @staticmethod
    def get_provider(name: str) -> LLMProvider:
        """Return a ready-to-use provider instance.

        Raises:
            LLMProviderError: if *name* is unknown or construction fails.
        """
        # A visitor's own key (request-scoped) takes precedence over whatever
        # the deployment is configured with.
        creds = current_credentials()
        if creds is not None:
            note_used(creds.provider, creds.model)
            return _make_for_credentials(creds)
        if require_user_key():
            raise LLMProviderError(
                "This site does not provide a built-in AI model: add your own API key to use one."
            )
        key = (name or "").strip().lower()
        note_used(key, None)
        builder = _PROVIDERS.get(key)
        if builder is None:
            supported = ", ".join(sorted(_PROVIDERS))
            raise LLMProviderError(
                f"Unknown LLM provider '{name}'. Supported: {supported}"
            )
        return builder()

    @staticmethod
    def get_default_provider() -> LLMProvider:
        """Return a provider selected by the ``LLM_PROVIDER`` env var.

        Falls back to ``"gemini"`` when the variable is unset or invalid.
        """
        name = os.environ.get("LLM_PROVIDER", _DEFAULT_PROVIDER).strip().lower()
        if name not in _PROVIDERS:
            logger.warning(
                "Unknown LLM_PROVIDER '%s', falling back to '%s'",
                name, _DEFAULT_PROVIDER,
            )
            name = _DEFAULT_PROVIDER
        return LLMFactory.get_provider(name)